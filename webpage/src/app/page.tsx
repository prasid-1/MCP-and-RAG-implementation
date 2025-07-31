"use client";
import React, { useCallback, useState } from "react";

export default function Home() {
  const [droppedFiles, setDroppedFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [message, setMessage] = useState("");

  const handleDragOver = useCallback(
    (event: React.DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      setIsDragging(true);
    },
    []
  );

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDragging(false);

    const files = event.dataTransfer.files;
    if (files.length > 0) {
      const newFiles = Array.from(files);
      setDroppedFiles((prevFiles) => [...prevFiles, ...newFiles]);
    }
  }, []);

  const uploadFiles = useCallback(async () => {
    if (droppedFiles.length === 0) {
      setMessage("No Files Provided");
      return;
    }

    setIsUploading(true);
    setMessage("");

    const formData = new FormData();
    droppedFiles.forEach((file) => {
      formData.append(`file`, file);
    });
    try {
      const backendUrl = "http://localhost:3001/upload-pdf";
      try {
        const response = await fetch(backendUrl, {
          method: "POST",
          // body: formData,
          body: formData,
        });
        if (!response.ok) {
          throw new Error("Failed to upload PDF file");
        }

        const result = await response.json();
        setMessage(`Files uploaded successfully: ${result.message}`);
        console.log("Files uploaded successfully:", result);
        setDroppedFiles([]);
      } catch (error) {
        console.error("Error connecting to backend:", error);
        return;
      }
    } catch (error) {
      console.error("Error uploading files:", error);
      setMessage("Error uploading files. Please try again.");
      return;
    } finally {
      setIsUploading(false);
      setDroppedFiles([]);
    }
  }, [droppedFiles]);
  return (
    <main>
      <div className="text-5xl font-bold text-center mt-30">
        <span className="">Customized</span>
        <span className="text-blue-500">AIresponses</span>
      </div>
      <div className="text-2xl font-semibold text-center mt-10">
        Upload your customization data files here
      </div>
      <div className="flex justify-center mt-10">
        <div className="flex flex-col items-center justify-center w-1/2">
          <label
            htmlFor="dropzone-file"
            className={`flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer bg-blue-50 hover:bg-blue-100 ${
              isDragging ? "border-blue-500" : "border-gray-400"
            } p-4`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              id="dropzone-file"
              type="file"
              className="hidden"
              onChange={(e) => {
                const files = e.target.files;
                if (files) {
                  const newFiles = Array.from(files);
                  setDroppedFiles((prevFiles) => [...prevFiles, ...newFiles]);
                }
              }}
            />
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <svg
                className="w-8 h-8 mb-4 text-gray-500 dark:text-gray-400"
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 20 16"
              >
                <path
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
                />
              </svg>
              <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
                <span className="font-semibold">Click to upload</span> or drag
                and drop
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                PDF (MAX. 50MB)
              </p>
            </div>
            {/* <input id="dropzone-file" type="file" className="hidden" /> */}
            {/* Display Dropped Files */}
            {droppedFiles.length > 0 && (
              <div className="mt-6">
                <h2 className="text-xl font-semibold text-gray-700 mb-3">
                  Dropped Files:
                </h2>
                <ul className="list-disc list-inside space-y-2 text-gray-600">
                  {droppedFiles.map((file, index) => (
                    <li
                      key={index}
                      className="bg-gray-50 p-2 rounded-md flex justify-between items-center"
                    >
                      <span className="truncate">{file.name}</span>
                      <span className="text-xs text-gray-400 ml-2">
                        ({(file.size / 1024).toFixed(2)} KB)
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </label>
          {droppedFiles.length > 0 && (
            <>
              <button
                onClick={uploadFiles}
                disabled={isUploading}
                className="mt-6 w-full bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-lg transition-colors duration-300 shadow-md"
              >
                {isUploading ? "processing..." : "Upload Files"}
              </button>
              <button
                onClick={() => setDroppedFiles([])}
                className="mt-2 w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition-colors duration-300 shadow-md"
              >
                Clear Files
              </button>
            </>
          )}
          {message && (
            <div className="mt-4 text-center text-green-500 font-semibold">
              {message}
            </div>
          )}
        </div>
      </div>

      <button
        onClick={() => (window.location.href = "/Prompt")}
        className="absolute left-1/2 transform -translate-x-1/2 mt-10 text-center w-1/4 bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition-colors duration-300 shadow-md"
      >
        Give Prompt
      </button>
    </main>
  );
}
