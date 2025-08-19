# Intro to Data Science Repository

Welcome! This repository showcases a collection of data‑science projects, machine‑learning models, and a personal portfolio website (Introductory Dara Science Course) . It is organized into several sections, each focusing on different aspects of data analysis, model building, and web development.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Running the Projects](#running-the-projects)
  - [Analysis Project (Streamlit)](#analysis-project-streamlit)
  - [Jupyter Notebooks](#jupyter-notebooks)
  - [Portfolio Website (Assignment‑01)](#portfolio-website-assignment‑01)
- [Contributing](#contributing)
- [License](#license)

## Overview
- **Analysis Project** – End‑to‑end data‑science workflow for Bitcoin price analysis, sentiment analysis on tweets, and predictive modeling (LSTM & Random Forest).  
- **Assignment‑01** – A static portfolio website built with HTML, CSS, and JavaScript.  
- **Assignment2 & Assignment3** – Additional notebooks demonstrating Python fundamentals and advanced concepts.

## Getting Started

### Prerequisites
- Python 3.11 (or any recent 3.x version)  
- Git (to clone the repo)  
- A modern web browser (for the portfolio site)

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/usman9522/Intro-to-Data-Science.git
cd Intro-to-Data-Science

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
.\venv\Scripts\activate    # Windows

# 3. Install Python dependencies for the analysis project
pip install -r Analysis_Project/requirements.txt
```

## Running the Projects

### Analysis Project (Streamlit)
The main entry point is `Analysis_Project/FINAL_IDS.py`. Launch the app with:

```bash
streamlit run Analysis_Project/FINAL_IDS.py
```

Open a browser at `http://localhost:8501` to explore interactive visualizations, tweet sentiment analysis, and Bitcoin price predictions.

### Jupyter Notebooks
Open any notebook using Jupyter or VS Code:

```bash
jupyter notebook Analysis_Project/IDSS.ipynb
jupyter notebook Assignment2/Assignment2.ipynb
jupyter notebook Assignment3/Assignment3.ipynb
```

### Portfolio Website (Assignment‑01)
The site is static HTML/CSS/JS. View it locally by opening `Assignment-01-/index.html` directly in a browser, or serve it with a simple HTTP server:

```bash
# Option 1: Open the file directly (macOS/Linux)
open Assignment-01-/index.html
# Windows
start Assignment-01-\index.html

# Option 2: Serve with Python's built‑in server
python -m http.server 8000 --directory Assignment-01-
# Then visit http://localhost:8000 in your browser
```

## Contributing
Contributions are welcome, especially for the **Analysis Project**. If you'd like to improve the data‑science workflow, add new visualizations, or enhance the models, please:

1. Fork the repository.  
2. Create a new branch for your changes.  
3. Submit a pull request with a clear description of what you have added or modified.

## License
This repository is provided for educational purposes.

---

*Happy coding!* 🚀
