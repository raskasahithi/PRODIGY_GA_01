# GPT-2 Text Generation Project

Welcome to the GPT-2 Text Generation project! This repository contains code for training and generating text using OpenAI's GPT-2 model. Whether you're fine-tuning existing models or creating your own text generation masterpieces, this project has you covered.

## Project Structure

- `train.py`: This script is used for training the GPT-2 model. You can fine-tune the pre-trained GPT-2 weights or train from scratch.
- `app.py`: The main application script for generating text using the trained GPT-2 model.
- `templates/`: Contains HTML templates for the frontend.
- `static/`: Place any static assets (CSS, images, etc.) here.

## Getting Started

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/your-username/PRODIGY_GA_01.git
    cd PRODIGY_GA_01
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Train the GPT-2 model (modify `train.py` as needed):

    ```bash
    python train.py
    ```

4. Run the web application:

    ```bash
    uvicorn app:app --reload
    ```

5. Access the app in your browser at `http://localhost:8000`.

## Usage

- Visit the web app and input a prompt to generate text using the trained GPT-2 model.
- Experiment with different prompts and see what creative output you get!

## Contributing

Contributions to this project are welcome! If you have ideas for improvements, new features, or enhancements, feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
