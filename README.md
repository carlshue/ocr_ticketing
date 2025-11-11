# ocr_ticketing

# Ticket OCR App

This project is a Flutter application that allows users to capture photos of tickets/receipts, send them to a server for processing, and receive structured ticket data for storage and analysis.

---

## Architecture Overview

```mermaid
flowchart TD
    A["Flutter App / User"] --> B["Capture ticket image"]
    B --> C["Send image to Server on Railway"]
    C --> D["Server Processing"]
    D --> D1["Image preprocessing - normalize, denoise"]
    D --> D2["Text extraction (OCR)"]
    D --> D3["Data parsing & structuring"]
    D --> D4["Optional analytics / enrichment"]
    D4 --> E["Structured ticket table (JSON)"]
    E --> F["Return processed data to Flutter App"]
    F --> G["Store ticket locally / display to user"]
