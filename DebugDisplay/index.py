from flask import Flask, send_from_directory
app = Flask("DebugDisplayServer")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def send_static(path):
    return send_from_directory("static", path)

if __name__ == "__main__":
     app.run(host="0.0.0.0", port=8080)