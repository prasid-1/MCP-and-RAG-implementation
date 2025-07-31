"use client";
export default async function UploadJava() {
  //create a dummy json file to upload
  const handleUpload = async () => {
    console.log("Uploading Java file...");
    const dataTosend = {
      name: "kobwe",
    };

    try {
      const response = await fetch("http://localhost:8080/users", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(dataTosend),
      });

      if (!response.ok) {
        throw new Error("Failed to upload Java file");
      }

      const result = await response.json();
      console.log("Upload successful:", result);
    } catch (error) {
      console.error("Error uploading Java file:", error);
    }
  };

  return (
    <div>
      <h1>Upload Java Files</h1>
      <button onClick={handleUpload}>Upload</button>
    </div>
  );
}
