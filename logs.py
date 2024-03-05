from flask import Flask
from src.logger import logging

# Create a Flask application instance
app = Flask(__name__)

 # Define a route for the root URL with GET and POST methods
@app.route('/', methods=['GET', 'POST'])
def index():
    logging.info("This is for testing our logging file")

    return "Welcome to Retail Customer Segmentation"

if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask application in debug mode on port 5000 by default