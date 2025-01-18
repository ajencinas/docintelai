# Project Title

Application to analyze and obtain insights from Rate Case Files and Utility Regulatory documents using LLMs

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- Add documents: Provides an easy way to add, manage and vectorize documents
- Query documents: Provides chatbot capability for documents, including query improvement and metadata filtering

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/ajencinas/docintelai.git
    ```
    
2. Navigate to project

    ```bash
    cd docintelai
    ```
    
3. Create virtual environment and add dependencies
    
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

1. Run streamlit

    ```bash
    streamlit run src/AgentFrontend.py
    ```

2. Open your browser and navigate to the URL provided by Streamlit (typically http://localhost:8501). 

## Contributing

Contributions are always welcome! To get started:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.


