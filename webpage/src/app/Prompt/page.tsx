"use client";

import { useState, useRef, useEffect } from "react";

export default function PromptPage() {
  const [promptInput, setPromptInput] = useState("");
  const [chatHistory, setChatHistory] = useState<string[]>([]);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatHistory]);

  function SendPrompt() {
    const input = document.getElementById("large-input") as HTMLInputElement;

    const PromptData = {
      prompt: `${input.value}`,
    };

    if (!input.value) {
      alert("Please enter a prompt.");
      return;
    }
    setChatHistory((prev) => [...prev, `loading...`]);

    try {
      fetch("http://localhost:3001/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(PromptData),
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Response from backend:", data);
          setChatHistory((prev) => [
            ...prev,
            `user: ${input.value}`,
            `AI: ${data.response}`,
          ]);
          input.value = ""; // Clear the input field
          const output = document.getElementById("output") as HTMLOutputElement;
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    } catch (error) {
      console.error("Error sending prompt:", error);
      const output = document.getElementById("output") as HTMLOutputElement;
      output.value = "Error sending prompt. Please try again.";
    }
  }

  return (
    <div className="mb-6 items-center flex justify-center flex-col">
      <div className="text-6xl font-bold text-center mt-20">
        <span className="">Prompt</span>
        <span className="text-blue-500">Handler</span>
      </div>
      <div
        id="output"
        className="mt-10 border border-gray-300 rounded-lg p-4 w-3/4 bg-blue-50 text-gray-700 space-y-2 overflow-y-auto"
      >
        {chatHistory.length === 0 ? (
          <p>Your prompt output will appear here.</p>
        ) : (
          chatHistory.map((line, index) => (
            <p
              key={index}
              className={line.startsWith("User:") ? "font-semibold" : ""}
            >
              {line}
            </p>
          ))
        )}
        <div ref={bottomRef} />
      </div>
      <div className="flex flex-row items-center justify-center w-3/4 mt-10">
        <input
          type="text"
          id="large-input"
          placeholder="Type your prompt here..."
          required
          className="block w-3/4 pt-5 pb-5 mt-5 text-gray-900 border border-gray-300 rounded-lg bg-blue-200 text-base focus:ring-blue-500 focus:border-blue-500"
        />
        <button
          className="mt-5 w-40 pt-5 pb-5 bg-blue-500 hover:bg-blue-600 text-white font-bold rounded-lg transition-colors duration-300 shadow-md"
          onClick={SendPrompt}
        >
          Submit
        </button>
      </div>
      <button
        onClick={() => (window.location.href = "/")}
        className=" mt-10 text-center w-1/4 bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition-colors duration-300 shadow-md"
      >
        Go to Home
      </button>
    </div>
  );
}
