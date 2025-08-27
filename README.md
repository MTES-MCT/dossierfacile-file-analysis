# DossierFacile File Analysis

This project is a microservice designed to analyze files for quality, with a primary focus on detecting blurry images. It operates as a consumer, listening to a RabbitMQ queue for analysis tasks.

## Analysis Pipeline

The service automates the following workflow upon receiving a message:

1.  **Message Consumption**: Listens to a specified AMQP (RabbitMQ) queue for incoming messages.
2.  **Data Retrieval**: Fetches file metadata from the database using the ID provided in the message.
3.  **File Download**: Downloads the file from its storage location (e.g., OVH, local).
4.  **Data Preparation**: 
    - If the file is a PDF, it's converted into a series of PNG images.
    - If the file is an image, it's used directly.
5.  **Blurry Analysis**:
    - Each image is analyzed to determine if it's blurry.
    - The analysis calculates the variance of the Laplacian of the image. A variance below a certain threshold indicates a blurry image.
    - Blank or nearly white images are identified and excluded from the analysis.
6.  **Result Storage**: The analysis result (especially the blurriest image's data in a multi-page document) is saved back to the database.

## Configuration

This project uses a `.env` file to manage environment variables. To get started, copy the example file:

```bash
cp .env.example .env
```

Then, edit the `.env` file to set your specific configuration for the following variables:

- `PROJECT_NAME`: A name for the project.
- `AMQP_URL`: The connection URL for the RabbitMQ server.
- `AMQP_USERNAME`: Username for RabbitMQ.
- `AMQP_PASSWORD`: Password for RabbitMQ.
- `AMQP_QUEUE_NAME`: The queue to listen to for analysis tasks.
- `ELASTIC_APM_*`: Variables for Elastic APM integration (optional).

## Running the Service

To run the service locally, ensure you have installed the dependencies and configured your `.env` file.

1.  **Install dependencies:**
    ```bash
    poetry install
    ```
2.  **Start the service:**
    ```bash
    poetry run python src/dossierfacile_file_analysis/main.py
    ```

## Running Tests

To run the tests, you first need to install the development dependencies:

```bash
poetry install --with dev
```

Then, you can run the tests using `pytest`:

```bash
poetry run pytest
```

## TESSERACT OCR : 
This project uses Tesseract OCR for text recognition in images. To install Tesseract on your system, follow the instructions below based on your operating system:
- **Ubuntu/Debian**:
  ```bash
  sudo apt-get install tesseract-ocr
  ```
- **macOS** (using Homebrew):
  ```bash
  brew install tesseract
  ```
  
The configuration of Tesseract use fra language by default. You need to install it from here : 
`https://github.com/tesseract-ocr/tessdata_fast/raw/main/fra.traineddata`
and put it inside your tesseract tessdata folder.

For Scalingo, we use a specific bash script to install the fra language during the startup.
`/bin/start_worker.sh`

Configuration : 
    - Tesseract OEM (OCR Engine Mode) : 1 (LSTM only), This mode is more accurate for most cases and faster.
    - PSM (Page Segmentation Mode) : 11 (Sparse text with OSD), This mode is suitable for images with sparse text and automatically detects the orientation and script of the text.