```text
handwriting_ai/               # handwriting model
│
├── data/
│   ├── raw/                  # original uploaded images
│   ├── processed/            # cleaned + deskewed images
│   ├── lines/                # segmented line images
│   ├── labels/               # text labels (if available)
│   ├── metadata/             # json mapping (image ↔ text)
│   └── user_data/            # per-user storage
│
├── models/
│   ├── ocr/                  # OCR models
│   ├── summarizer/           # summarization models
│   └── handwriting/          # style/font models
│
├── src/
│   ├── preprocessing/        # image cleaning, deskew
│   ├── segmentation/         # line segmentation
│   ├── ocr/                  # OCR inference + training
│   ├── correction/           # auto-correction logic
│   ├── summarization/        # summary logic
│   ├── handwriting/          # style generation
│   └── utils/                # helpers
│
├── pipelines/
│   ├── train_pipeline.py
│   ├── inference_pipeline.py
│   └── full_pipeline.py
│
├── outputs/
│   ├── text/                 # extracted text
│   ├── summaries/            # generated summaries
│   └── handwriting/          # final styled outputs
│
├── configs/
│   └── config.yaml
│
├── logs/
│
└── main.py
```