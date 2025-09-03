"use client";

import { useRef, useState } from "react";
import { formatWithOptions } from "util";
export default function UploadPDF() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [word, setWord] = useState("");
  const [definition, setDefinition] = useState("");
  const [example, setExample] = useState("");
  const [mnemonic, setMnemonic] = useState("");
  const [fileName, setFileName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [lastUploadedFile, setLastUploadedFile] = useState<string | null>(null);
  const [alreadyUploaded, setAlreadyUploaded] = useState(false);

  const handleUpload = async () => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      alert("Please select image file to upload.");
      return;
    }

    const fileId = `${file.name}_${file.lastModified}`;
    // if (fileId === lastUploadedFile) {
    //   alert("This file was already uploaded.");
    //   return;
    // }

    setUploading(true);
    setAlreadyUploaded(false);
    setLastUploadedFile(fileId);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:3001/ocr-image", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload images");
      }

      const result = await response.json();
      console.log("Upload successful:", result);
    } catch (error) {
      console.error("Error uploading images:", error);
    } finally {
      setUploading(false);
    }
  };

  const clearForm = () => {
    setWord("");
    setDefinition("");
    setExample("");
    setMnemonic("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    setAlreadyUploaded(false);
  };

  const getAutoFillData = async () => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      alert("Please select an image file to upload.");
      return;
    }
    const formData = new FormData();
    formData.append("filename", file.name);

    try {
      const response = await fetch("http://localhost:3001/getFromData", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        alert("failed to fetch");
        throw new Error("Failed to fetch");
      }

      // const raw = await response.text(); //get the raw text response
      // console.log("Raw response:", raw);

      const result = await response.json(); //parse the JSON response
      console.log("Received JSON:", result);

      if (Array.isArray(result) && result.length > 0) {
        const firstEntry = result[0];
        setWord(firstEntry.word || "");
        setDefinition(firstEntry.definition || "");
        setExample(firstEntry.example || "");
        setMnemonic(firstEntry.mnemonic || "");
      } else {
        alert("No valid data found in JSON.");
      }
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
      {/* Upload Box */}
      <div className="bg-white rounded-2xl shadow-md p-10 w-full max-w-md text-center mb-10">
        <h1 className="text-2xl font-bold text-gray-800 mb-6">
          Upload an Image
        </h1>
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          onChange={() => setAlreadyUploaded(false)} // Reset if a new file is selected
          className="block w-full text-sm text-gray-600
                     file:mr-4 file:py-2 file:px-4
                     file:rounded-full file:border-0
                     file:text-sm file:font-semibold
                     file:bg-blue-100 file:text-blue-700
                     hover:file:bg-blue-200 mb-6"
        />
        <button
          onClick={handleUpload}
          disabled={uploading || alreadyUploaded}
          className={`${
            uploading
              ? "bg-gray-400 cursor-not-allowed"
              : alreadyUploaded
              ? "bg-green-600"
              : "bg-blue-600 hover:bg-blue-700"
          } text-white font-semibold py-2 px-6 rounded-xl transition duration-200`}
        >
          {uploading
            ? "Uploading..."
            : alreadyUploaded
            ? "Uploaded"
            : "Upload Image"}
        </button>
      </div>

      {/* Word Entry Form */}
      <div className="bg-white rounded-2xl shadow-md p-10 w-full max-w-md">
        <h2 className="text-xl font-semibold text-gray-800 mb-6">
          Enter Word Details
        </h2>
        <form className="flex flex-col space-y-4">
          <div>
            <label className="block mb-1 font-medium text-gray-700">
              Word:
            </label>
            <input
              type="text"
              value={word}
              onChange={(e) => setWord(e.target.value)}
              className="w-full border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="e.g., Ubiquitous"
            />
          </div>
          <div>
            <label className="block mb-1 font-medium text-gray-700">
              Definition:
            </label>
            <textarea
              value={definition}
              onChange={(e) => setDefinition(e.target.value)}
              className="w-full border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="e.g., Present everywhere"
            />
          </div>
          <div>
            <label className="block mb-1 font-medium w-40 text-gray-700">
              Example:
            </label>
            <textarea
              value={example}
              onChange={(e) => setExample(e.target.value)}
              className="w-full border rounded-lg px-4 py-2 h-25 focus:outline-none focus:ring-2 focus:ring-blue-500 auto-fill:bg-blue-50"
              placeholder="e.g., Smartphones are ubiquitous these days."
            />
          </div>
          <div>
            <label className="block mb-1 font-medium text-gray-700">
              Mnemonic:
            </label>
            <textarea
              value={mnemonic}
              onChange={(e) => setMnemonic(e.target.value)}
              className="w-full border rounded-lg px-4 py-2 h-25 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="e.g., Sounds like 'You big with us'"
            />
          </div>
          <button
            type="button"
            onClick={getAutoFillData}
            className="bg-blue-600 text-white font-semibold py-2 px-6 rounded-xl hover:bg-blue-700 transition duration-200"
          >
            Auto Fill
          </button>
          <button
            type="submit"
            className="mt-4 bg-green-600 text-white font-semibold py-2 px-6 rounded-xl hover:bg-green-700 transition duration-200"
            onClick={(e) => {
              e.preventDefault();
              console.log("Submitted:", {
                word,
                definition,
                example,
                mnemonic,
              });
            }}
          >
            Submit Word
          </button>
          <button
            type="button"
            onClick={clearForm}
            className="bg-red-600 text-white font-semibold py-2 px-6 rounded-xl hover:bg-red-700 transition duration-200"
          >
            Clear
          </button>
        </form>
      </div>
    </div>
  );
}
