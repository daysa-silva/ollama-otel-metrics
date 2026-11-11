package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"time"
	"strings"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)


// Structs to parse JSON responses from Ollama
type generateResponse struct {
	Model              string      `json:"model"`
	Done               *bool       `json:"done,omitempty"`
	Response           string      `json:"response,omitempty"`
	PromptEvalCount    *int        `json:"prompt_eval_count,omitempty"`
	EvalCount          *int        `json:"eval_count,omitempty"`
	TotalDuration      *int64      `json:"total_duration,omitempty"`
	EvalDuration       *int64      `json:"eval_duration,omitempty"`
	PromptEvalDuration *int64      `json:"prompt_eval_duration,omitempty"`
	DoneReason         interface{} `json:"done_reason,omitempty"`
}

type psResponse struct {
	Models []struct {
		Name      string `json:"name"`
		ID        string `json:"id"`
		Size      int64  `json:"size"`      // Size in bytes
		Processor string `json:"processor"`
		Until     string `json:"until"`
	} `json:"models"`
}

type modelsResponse struct {
	Models []struct {
		Name    string `json:"name"`
		Model   string `json:"model"`
		Size    int64  `json:"size"`
		Digest  string `json:"digest"`
		Details struct {
			ParentModel       string   `json:"parent_model"`
			Format            string   `json:"format"`
			Family            string   `json:"family"`
			Families          []string `json:"families"`
			ParameterSize     string   `json:"parameter_size"`
			QuantizationLevel string   `json:"quantization_level"`
		} `json:"details"`
		ExpiresAt string `json:"expires_at"`
		SizeVRAM  int64  `json:"size_vram"`
	} `json:"models"`
}

// Custom metrics handler that refreshes model data before serving metrics
func metricsHandler(w http.ResponseWriter, r *http.Request) {
	upstreamAddr := getUpstreamAddr()

	// Refresh model information from /api/ps before serving metrics
	if resp, err := upstreamClient.Get(upstreamAddr + "/api/ps"); err == nil {
		var ps psResponse
		data, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if jsonErr := json.Unmarshal(data, &ps); jsonErr == nil {
			loadedModelsGauge.Set(float64(len(ps.Models)))
			loadedModelInfo.Reset()
			modelRAMUsage.Reset() // Reset RAM usage before updating
			
			// Update model metrics directly from ps response
			for _, m := range ps.Models {
				modelName := ensureModelTag(m.Name)
				loadedModelInfo.WithLabelValues(modelName).Set(1)
				
				// Convert size from bytes to megabytes
				ramMB := float64(m.Size) / (1024 * 1024)
				modelRAMUsage.WithLabelValues(modelName).Set(ramMB)
			}
			log.Printf("Refreshed metrics data: %d models loaded", len(ps.Models))
		} else {
			log.Printf("ERROR parsing /api/ps response during metrics refresh: %v", jsonErr)
		}
	} else {
		log.Printf("ERROR refreshing model metrics: %v", err)
	}
	
	// Serve metrics using the standard Prometheus handler
	promhttp.Handler().ServeHTTP(w, r)
}

// All other paths -> proxy handler
func proxyHandler(w http.ResponseWriter, r *http.Request) {
	upstreamAddr := getUpstreamAddr()
	start := time.Now()
	if r.URL.Path == "/metrics" {
		// Should be handled above; just in case
		promhttp.Handler().ServeHTTP(w, r)
		return
	}

	// Read request body (if any) for forwarding and possibly to inspect model parameter
	var bodyBytes []byte
	if r.Body != nil {
		bodyBytes, _ = io.ReadAll(r.Body)
		r.Body.Close()
	}

	// Build upstream request
	targetURL := upstreamAddr + r.URL.Path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}
	reqUp, err := http.NewRequest(r.Method, targetURL, bytes.NewReader(bodyBytes))
	if err != nil {
		http.Error(w, "Bad request", http.StatusInternalServerError)
		log.Printf("ERROR creating upstream request: %v", err)
		return
	}
	// Copy original headers
	for name, vals := range r.Header {
		if name == "Host" {
			continue // skip Host header
		}
		for _, v := range vals {
			reqUp.Header.Add(name, v)
		}
	}
	if reqUp.Header.Get("Content-Type") == "" && len(bodyBytes) > 0 {
		reqUp.Header.Set("Content-Type", "application/json")
	}

	respUp, err := upstreamClient.Do(reqUp)
	if err != nil {
		http.Error(w, "Upstream error", http.StatusBadGateway)
		log.Printf("ERROR forwarding to upstream: %v", err)
		return
	}
	defer respUp.Body.Close()

	// Forward status code and headers (except hop-by-hop headers)
	for name, vals := range respUp.Header {
		switch strings.ToLower(name) {
		case "transfer-encoding", "content-length":
			continue
		}
		for _, v := range vals {
			w.Header().Add(name, v)
		}
	}
	w.WriteHeader(respUp.StatusCode)

	// Variables for metrics/logging
	var modelName string
	var promptCount, generatedCount int
	var evalDurationNs int64

	if r.URL.Path == "/api/generate" || r.URL.Path == "/api/chat" {
		// Handle streaming JSON response for generate/chat
		var buf bytes.Buffer
		// MultiWriter to write to client output and buffer simultaneously
		streamWriter := io.MultiWriter(w, &buf)
		flusher, _ := w.(http.Flusher)
		// Stream copy loop
		chunk := make([]byte, 1024)
		for {
			n, readErr := respUp.Body.Read(chunk)
			if n > 0 {
				// Write chunk to both client and buffer
				if _, writeErr := streamWriter.Write(chunk[:n]); writeErr != nil {
					log.Printf("WARNING: client write error: %v", writeErr)
					break
				}
				if flusher != nil {
					flusher.Flush() // flush chunk to client
				}
			}
			if readErr != nil {
				// Break on EOF or error
				if readErr != io.EOF {
					log.Printf("ERROR reading upstream: %v", readErr)
				}
				break
			}
		}

		// Fix the JSON before parsing
		bufData := fixDoneReason(buf.Bytes())

		// First, try to parse as models list response
		var modelsResp modelsResponse
		if err := json.Unmarshal(bufData, &modelsResp); err == nil && len(modelsResp.Models) > 0 {
			// This is a models list response, not a typical generate response
			for _, model := range modelsResp.Models {
				modelName = ensureModelTag(model.Model)
				modelRAMUsage.WithLabelValues(modelName).Set(float64(model.SizeVRAM) / (1024 * 1024))
			}
			log.Printf("Received models list response with %d models", len(modelsResp.Models))
			// No token counts for models list response
		} else {
			// Parse lines of JSON objects as standard generate responses
			for _, line := range bytes.Split(bufData, []byte{'\n'}) {
				if len(line) == 0 {
					continue
				}

				var res generateResponse
				if err := json.Unmarshal(line, &res); err != nil {
					log.Printf("WARNING: Failed to parse JSON: %v", err)
					continue
				}

				if res.Model != "" {
					modelName = ensureModelTag(res.Model)
				}

				// If this chunk signals done, capture metrics fields
				if res.Done != nil && *res.Done {
					if res.PromptEvalCount != nil {
						promptCount = *res.PromptEvalCount
					}
					if res.EvalCount != nil {
						generatedCount = *res.EvalCount
					}
					if res.EvalDuration != nil {
						evalDurationNs = *res.EvalDuration
					}
				}
			}
		}
	} else if r.URL.Path == "/api/ps" {
		// Intercept /api/ps to update loaded models metrics
		bodyData, _ := io.ReadAll(respUp.Body)
		w.Write(bodyData) // write response to client
		var ps psResponse
		if err := json.Unmarshal(bodyData, &ps); err == nil {
			loadedModelsGauge.Set(float64(len(ps.Models)))
			loadedModelInfo.Reset()
			modelRAMUsage.Reset() // Reset RAM usage before updating
			
			// Update model metrics directly from ps response
			for _, m := range ps.Models {
				modelName := ensureModelTag(m.Name)
				loadedModelInfo.WithLabelValues(modelName).Set(1)
				
					// Convert size from bytes to megabytes
				ramMB := float64(m.Size) / (1024 * 1024)
				modelRAMUsage.WithLabelValues(modelName).Set(ramMB)
				log.Printf("Model %s RAM usage: %.2f MB", modelName, ramMB)
			}
		} else {
			log.Printf("ERROR parsing /api/ps response: %v", err)
		}
		// We can return here since request is fully handled
		return
	} else {
		// Other endpoints (e.g. /api/tags, /api/pull) - just copy through
		io.Copy(w, respUp.Body)
		// Try to get model name from request (if JSON body has "model")
		if len(bodyBytes) > 0 {
			var reqJson map[string]interface{}
			if err := json.Unmarshal(bodyBytes, &reqJson); err == nil {
				if m, ok := reqJson["model"].(string); ok {
					modelName = ensureModelTag(m)
				}
			}
		}
		// No token metrics for non-generate endpoints
	}

	// Update Prometheus metrics if applicable
	if modelName != "" {
		if promptCount > 0 {
			promptTokens.WithLabelValues(modelName).Add(float64(promptCount))
		}
		if generatedCount > 0 {
			generatedTokens.WithLabelValues(modelName).Add(float64(generatedCount))
		}
		durationSec := time.Since(start).Seconds()
		endpoint := strings.TrimPrefix(r.URL.Path, "/api/")
		if endpoint == "" {
			endpoint = r.URL.Path
		}
		requestDuration.WithLabelValues(endpoint, modelName).Observe(durationSec)
		if generatedCount > 0 && evalDurationNs > 0 {
			// evalDurationNs is in nanoseconds
			secondsPerToken := float64(evalDurationNs) / 1e9 / float64(generatedCount)
			timePerToken.WithLabelValues(modelName).Observe(secondsPerToken)
		}
		// Removed the redundant /api/ps call for /api/generate and /api/chat endpoints
		// Model metrics are now refreshed on every /metrics request
	}

	// Logging
	duration := time.Since(start).Seconds()
	if modelName != "" && (promptCount > 0 || generatedCount > 0) {
		log.Printf("%s %s -> model=%s prompt_tokens=%d generated_tokens=%d duration=%.2fs",
			r.Method, r.URL.Path, modelName, promptCount, generatedCount, duration)
	} else {
		log.Printf("%s %s -> status=%d duration=%.2fs", r.Method, r.URL.Path, respUp.StatusCode, duration)
	}
}