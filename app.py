from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_placement():
    gp = float(request.form.get('GP'))
    minyrs = float(request.form.get('MIN'))
    pts = float(request.form.get('PTS'))
    fgm = float(request.form.get('FGM'))
    fga = float(request.form.get('FGA'))
    fg = float(request.form.get('FG'))
    threepmade = float(request.form.get('ThreePMade'))
    threepa = float(request.form.get('ThreePA'))
    threep = float(request.form.get('ThreeP'))
    ftm = float(request.form.get('FTM'))
    fta = float(request.form.get('FTA'))
    ft = float(request.form.get('FT'))
    oreb = float(request.form.get('OREB'))
    dreb = float(request.form.get('DREB'))
    reb = float(request.form.get('REB'))
    ast = float(request.form.get('AST'))
    stl = float(request.form.get('STL'))
    blk = float(request.form.get('BLK'))
    tov = float(request.form.get('TOV'))

    # prediction
    result = model.predict(np.array([gp, minyrs, pts, fgm, fga, fg, threepmade, threepa, threep, ftm, fta, ft, oreb, dreb, reb, ast, stl, blk, tov]).reshape(1, 19))
    # result = model.predict(np.array([58, 11.6, 5.7, 2.3, 5.5, 42.6, 0.1, 0.5, 22.6, 0.9, 1.3, 68.9, 1, 0.9, 1.9, 0.8, 0.6, 0.1, 1]).reshape(1, 19))
    # result = model.predict(np.array([35, 26.9, 7.2, 2, 6.7, 29.6, 0.7, 2.8, 23.5, 2.6, 3.4, 76.5, 0.5, 2, 2.4, 3.7, 1.1, 0.5, 1.6]).reshape(1, 19))
    if result[0] == 1:
        result = 'Prediction says that the player will last five years in the league'
    else:
        result = 'Prediction says that the player will not last five years in the league'

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
