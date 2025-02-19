# Project Title

Application to analyze and obtain insights from Rate Case Files and Utility Regulatory documents using LLMs

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- FrontEnd: Provides easy way to query large collections of regulatory files
- BackEnd: 
    - Handles storage, metadata search, and filtering
    - Provides functionality to scrape PUC websites

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

TBC

## Azure configuration

Overall architecture is as follows:
1. Documents stored in Azure Blob
2. Vector FAISS stored in Azure Blob
3. Document overview (summaries, summary vectors, plus metadata) stored in Azure Cosmos vCore to enable vector search
4. Chats stored in Azure Cosmos MongoDB
5. Redis used to quickly process metadata search

For questions please email juanencinasmain@gmail.com

## Contributing

Contributions are always welcome! To get started:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## Notes

1. Package conflicts
	
	async-timeout==5.0.1 should be excluded from requirements.txt


