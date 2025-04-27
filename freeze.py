from flask_frozen import Freezer
from app import app

freezer = Freezer(app)

# Add URL generator for predict endpoint


@freezer.register_generator
def predict():
    yield {"path": "/predict"}


if __name__ == '__main__':
    freezer.freeze()
