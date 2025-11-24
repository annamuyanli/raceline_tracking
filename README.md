# Documentation

## Software Requirements

1. Python 3.10
2. Matplotlib
3. Numpy

## To run

```bash
python3 main.py ./racetracks/Montreal.csv ./racetracks/Montreal_raceline.csv
```

## Virtual environment

It is recommended to use a virtual environment:

```bash
cd raceline_tracking
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## To design controller

Edit `controller.py` to write controller. Other files can be edited, but with discretion.
