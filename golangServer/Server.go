package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
)

type User struct {
	Name string `json:"name"`
}

var userCache = make(map[int]User)

var cacheMutex = sync.RWMutex{}

func main() {
	mux := http.NewServeMux()

	mux.HandleFunc("/", handleRoot)

	mux.HandleFunc("POST /users", createUser)
	mux.HandleFunc("GET /users/{id}", getUser)
	mux.HandleFunc("DELETE /users/{id}", deleteUser)

	mux.HandleFunc("POST /upload-pdf", uploadPDFHandler)

	mux.HandleFunc("POST /query", queryHandler)

	fmt.Println("Starting server on :3001")
	http.ListenAndServe(":3001", mux)
}

func queryHandler(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var data struct {
		Prompt string `json:"prompt"`
	}

	err := json.NewDecoder(r.Body).Decode(&data)
	if err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	cmd := exec.Command("python", "RAG/queryData.py", data.Prompt)
	output, err := cmd.CombinedOutput()
	if err != nil {
		http.Error(w, "Failed to execute Python script: "+err.Error(), http.StatusInternalServerError)
		return
	}

	response := string(output)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"response": response,
	})
}

func uploadPDFHandler(w http.ResponseWriter, r *http.Request) {

	enableCORS(w)

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if err := r.ParseMultipartForm(50 << 20); err != nil { // 50 MB limit for form data in memory
		http.Error(w, "Error parsing multipart form: "+err.Error(), http.StatusBadRequest)
		return
	}
	// "file" is the name of the input field
	// make sure your HTML form has an input with name="file"
	// else die
	file, handler, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Error retrieving file from form data", http.StatusBadRequest)
		return
	}
	defer file.Close()

	uploadDir := "./uploads"
	if err := os.MkdirAll(uploadDir, os.ModePerm); err != nil {
		http.Error(w, "Error creating upload directory", http.StatusInternalServerError)
		return
	}
	// Accept any file name, but sanitize it to avoid directory traversal
	// filename := filepath.Base(handler.Filename)
	// Optionally, you can use filepath.Base to strip directory info
	// filename := filepath.Base(handler.Filename)

	filename := filepath.Join(uploadDir, filepath.Base(handler.Filename))

	dst, err := os.Create(filename)
	if err != nil {
		http.Error(w, "Error creating file on server", http.StatusInternalServerError)
		return
	}
	defer dst.Close()

	// Copy the uploaded file content to the new file
	if _, err := io.Copy(dst, file); err != nil {
		http.Error(w, "Error saving file to server", http.StatusInternalServerError)
		return
	}

	cmd := exec.Command("python", "RAG/populateDatabase.py", "--reset")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		http.Error(w, "Failed to execute Python script: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"message":  "File uploaded successfully",
		"filename": filename,
	})
}

func handleRoot(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	fmt.Fprintf(w, "Backend server is running. Use the /upload-pdf endpoint to upload files.")
}

func createUser(w http.ResponseWriter, r *http.Request) {

	enableCORS(w)

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	var user User

	err := json.NewDecoder(r.Body).Decode(&user)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if user.Name == "" {
		http.Error(w, "Name is required", http.StatusBadRequest)
		return
	}

	cacheMutex.Lock()
	userCache[len(userCache)+1] = user
	cacheMutex.Unlock()

	w.WriteHeader(http.StatusNoContent)
}

// Stub for getUser handler
func getUser(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	http.Error(w, "Not implemented", http.StatusNotImplemented)
}

// Stub for deleteUser handler
func deleteUser(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	http.Error(w, "Not implemented", http.StatusNotImplemented)
}

var allowedOrigin = "http://localhost:3000"

func enableCORS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE, PUT")
	w.Header().Set("Access-Control-Allow-Origin", allowedOrigin)
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
}
