from flask import Flask

app = Flask(__name__)
# App config.
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '74c1112c-d16f-446c-9b6f-ee3315b7ec8b'


@app.route('/')
def index():
    return 'hello, world

#lancement de l'application
if __name__ == "__main__":
    app.run(debug=True)

#if __name__ == '__main__':
    
 #   port = os.environ.get("Port", 5000)
 #   app.run(debug=True, host="0.0.0.0", port=port)
    

    
