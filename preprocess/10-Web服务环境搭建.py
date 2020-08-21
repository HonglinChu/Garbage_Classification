from flask import Flask

#通过curl http://...... 来访问app
app=Flask(__name__)

@app.route('/')#  
def hello():
    return 'hello world'

if  __name__=='__main__':
    app.run()   