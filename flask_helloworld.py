# Importujemy potrzebne moduły
from flask import Flask, render_template_string, render_template

# Inicjalizujemy aplikację Flask
app = Flask(__name__)

# Tworzymy zmienną zawierającą napis "Witaj świecie"
hello = "Witaj świecie!"

# Definiujemy trasę "/"
@app.route('/')
def hello_world():
    # Przekazujemy zmienną 'hello' do szablonu, który zostanie wyświetlony
    return render_template('helloworld_index.html', hello=hello)
    #return render_template_string('<h1> {{ hello }} </h1>', hello=hello)


# Uruchamiamy aplikację
if __name__ == '__main__':
    app.run(port=8080, debug=True)