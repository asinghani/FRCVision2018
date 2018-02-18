from flask import Flask
app = Flask("DebugDisplayServer")

@app.route("/")
def hello():
    return "<button onclick=\"window.location = '/page2';\" style=\"font-size:50px\">Test 1</button>"

@app.route("/page2")
def hello2():
    return "<button onclick=\"window.location = '/';\" style=\"font-size:50px\">Test 2</button>"

if __name__ == "__main__":
     app.run(host="0.0.0.0", port=8080)