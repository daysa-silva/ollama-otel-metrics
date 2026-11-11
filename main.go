package main

import (
	"context"
	"errors"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"regexp"
	"strings"
	"time"
	
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)



var upstreamClient = &http.Client{
	// No global timeout; model inference may take a long time
	Timeout: 0,
}

// fixDoneReason processes JSON data to handle the done_reason field that might be a number or string
func fixDoneReason(data []byte) []byte {
	re := regexp.MustCompile(`"done_reason":(\d+)`)
	return re.ReplaceAll(data, []byte(`"done_reason":"$1"`))
}

// ensureModelTag adds ":latest" to model names that don't have a tag
func ensureModelTag(modelName string) string {
	if modelName == "" {
		return modelName
	}
	if !strings.Contains(modelName, ":") {
		return modelName + ":latest"
	}
	return modelName
}

func getUpstreamAddr() string {
	upstreamAddr := os.Getenv("OLLAMA_HOST")

	if upstreamAddr == "" {
		upstreamAddr = "http://localhost:11434"
	}

	return upstreamAddr
}

func main() {
	// Handle SIGINT (CTRL+C) gracefully.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Set up OpenTelemetry.
	otelShutdown, err := setupOTelSDK(ctx)
	if err != nil {
		log.Fatalln(err)
	}
	// Handle shutdown properly so nothing leaks.
	defer func() {
		err = errors.Join(err, otelShutdown(context.Background()))
	}()

	// Configure upstream Ollama host and local port from environment
	upstreamAddr := getUpstreamAddr()

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Starting Ollama proxy sidecar, forwarding to %s", upstreamAddr)

	// Start HTTP server.
	listenAddr := ":" + port
	srv := &http.Server{
		Addr:         listenAddr,
		BaseContext:  func(net.Listener) context.Context { return ctx },
		ReadTimeout:  time.Second,
		WriteTimeout: 10 * time.Second,
		Handler:      newHTTPHandler(),
	}
	srvErr := make(chan error, 1)
	go func() {
		log.Printf("Listening on %s", listenAddr)
		srvErr <- srv.ListenAndServe()
	}()

	// Wait for interruption.
	select {
	case err = <-srvErr:
		// Error when starting HTTP server.
		if err != nil {
			log.Fatal(err)
		}
	case <-ctx.Done():
		// Wait for first CTRL+C.
		// Stop receiving signal notifications as soon as possible.
		stop()
	}

	// When Shutdown is called, ListenAndServe immediately returns ErrServerClosed.
	err = srv.Shutdown(context.Background())
	if err != nil {

		// Error when starting HTTP server.
		log.Fatal(err)
	}
}

func newHTTPHandler() http.Handler {
	mux := http.NewServeMux()

	// Register handlers.
	mux.Handle("/", http.HandlerFunc(proxyHandler))

	// Add HTTP instrumentation for the whole server.
	handler := otelhttp.NewHandler(mux, "/")
	return handler
}