# Front-end model viewer

## Run

```powershell
cd d:\FYP
pip install -r Front_end_model\requirements.txt
streamlit run Front_end_model\app.py
```

## Features

- Choose any `.pt` checkpoint from `d:\FYP\checkpoints`
- Upload an image for inference
- Show original image, predicted mask, and overlay
- Show pixel ratio by predicted class
