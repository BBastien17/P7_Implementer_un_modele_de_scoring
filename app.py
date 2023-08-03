from flask import Flask

app = Flask(__name__)


@app.route('/')
def index():
    return 'hello, world


if __name__ == '__main__':
    
    port = os.environ.get("Port", 5000)
    app.run(debug=False, host="0.0.0.0", port=port)
    
