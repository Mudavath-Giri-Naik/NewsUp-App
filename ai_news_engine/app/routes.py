from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from app.vision import extract_articles_from_images
import os
import shutil

router = APIRouter()


@router.post("/upload_folder")
async def upload_folder(file: UploadFile = File(...)):
    # Temporary folder to save the uploaded file
    temp_folder = "temp_folder"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    try:
        # Save the uploaded zip file
        file_path = os.path.join(temp_folder, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract folder name and use it for csv file
        folder_name = file.filename.split(".")[0]
        image_folder_path = os.path.join(temp_folder, folder_name)

        # Unzip the file
        if file.filename.endswith(".zip"):
            shutil.unpack_archive(file_path, image_folder_path)
        else:
            raise HTTPException(status_code=400, detail="Uploaded file is not a zip file.")

        # Process and generate CSV
        csv_file = f"{folder_name}.csv"
        csv_path = os.path.join("uploads", csv_file)

        # Make sure uploads folder exists
        os.makedirs("uploads", exist_ok=True)

        extract_articles_from_images(image_folder_path, csv_path)

        # Return response with downloadable URL
        return JSONResponse(content={
            "status": "success",
            "csv_file": csv_file,
            "download_url": f"/download/{csv_file}"
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)


@router.get("/download/{csv_filename}")
def download_csv(csv_filename: str):
    csv_path = os.path.join("uploads", csv_filename)
    if os.path.exists(csv_path):
        return FileResponse(
            path=csv_path,
            filename=csv_filename,
            media_type="text/csv"
        )
    raise HTTPException(status_code=404, detail="CSV file not found")
