

<p>Desarrollado por:<b> Insight ML</b></p>


<div>
    <b>Integrantes</b> 
    <ul>
        <li>Renato Castro Cruz</li>
        <li>Cristian Lazo Quispe</li>
    </ul>
</div>


<div><b>Descripción del reto</b></div>

<div>
    <p>El reto consiste en desarrollar un algoritmo que prediga la demanda de un modelo en un punto de venta de acuerdo a su gama semanalmente, sin usar herramientas, aplicaciones y/o API de paga y además de no generar gastos. <b>Esto es una restricción muy importante</b></p>
</div>


<div><b>Descripción del dataset</b></div>
<div>
    Se va a predecir a nivel de semana y se tiene los siguientes archivos:

<div>Files
    <ul>
        <li>train.csv - Data de entrenamiento. Desde la semana 1 (17/05/2021-23/05/2021) hasta la semana 50 (25/04/2022- 1/05/2022)</li>
        <li>test.csv - Data de test. Desde la semana 51 (2/05/2022-8/05/2022) hasta la semana 60 (4/07/2022-10/07/2022)</li>
        <li>test_sample.csv - Un ejemplo de submission en el formato correcto</li>
    </ul>
</div>
    
    
<div>ColumnsColumns
    <ul>
        <li>Z_MARCA - Marca del equipoZ_MARCA - Marca del equipo</li>
        <li>Z_GAMA - Gama del equipoZ_GAMA - Gama del equipo</li>
        <li>Z_MODELO - ModeloZ_MODELO - Modelo</li>
        <li>Z_DEPARTAMENTO - Departamento del Perú del punto de venta</li>
        <li>Z_PUNTO_VENTA - Punto de venta a abastecer</li>
        <li>SEMANA_XX - Semana de abastecimiento de acuerdo con el modelo, punto de venta y gama</li>
    </ul>
</div>
    
<div><b>Target</b></div>

<div>
    <p>Se pide determinar la demanda semanal de un modelo en un punto de venta de acuerdo a su gama.</p>
</div>

<div><b>Evaluación</b></div>

<div>
    <p>Se evaluará a los participantes en función del RMSE, de menor a mayor. Durante la competencia se reportarán resultados sobre un subconjunto del set de evaluación (50%), pero los resultados finales se calcularán sobre los datos restantes (50%)</p>
</div>

<div><b>Formato del Kaggle Submission</b></div>

<div>
    <p>Para todos los participantes, los archivos subidos deberán tener las siguientes columnas: ID yDemanda. Tener en cuenta que el valor de "Demanda" hace referencia al valor forecasteado. Al mismo tiempo, la estructura de la variable "ID" es Modelo|Punto de venta|Gama|# Semana.</p>
</div>

<div>   
El formato del archivo será el siguiente:

<pre><code>ID,Demanda
XXXX|PDV1|XXXX|Semana 55,1
XXXX|PDV1|XXXX|Semana 56,2
XXXX|PDV1|XXXX|Semana 57,3
</code></pre>

</div>
<p></p>
<div>
    <b>Lenguajes de Programación, Librerías, Frameworks a usar</b>
</div>
<p></p>
<div>
    Lenguajes de Programación:
    <ul>
        <li><a href='https://www.anaconda.com/'>Anaconda - Python</a></li>
        <li><a href='https://www.javascript.com/'>Javascript</a></li>
    </ul>
</div>

<div>
    Frameworks y librerías:
    <ul>
        <li><a href='https://scikit-learn.org/stable/'>Scikit-learn</a>, <a href='https://pandas.pydata.org/'>Pandas</a>, <a href='https://numpy.org/'>Numpy</a></li>
        <li><a href='https://lightgbm.readthedocs.io/en/latest/index.html'>LightGBM</a>, <a href='https://xgboost.ai/'>XgBoost</a>, <a href='https://catboost.ai/'>CatBoost</a></li>
        <li><a href='http://seaborn.pydata.org/'>Seaborn</a>, <a href='https://plotly.com/python/'>Plotly</a></li>
        <li><a href='https://facebook.github.io/prophet/'>Prophet</a></li>
        <li><a href='https://pytorch.org/'>PyTorch</a></li>
    </ul>
</div>


<div><b>Recursos que nos ayudaron en el desarrollo del proyecto:</b></div>

<div>
    Kaggle Competitions:
    <ul>
        <li><a href='https://www.kaggle.com/competitions/m5-forecasting-accuracy'>M5 Forecasting Accuracy 2020</a></li>
        <li><a href='https://www.kaggle.com/competitions/store-sales-time-series-forecasting'>Store Sales - Time Series Forecasting</a></li>
    </ul>
</div>

<div>
    Videos:
    <ul>
        <li><a href='https://www.youtube.com/watch?v=VYpAodcdFfA'>Two Effective Algorithms for Time Series Forecasting</a></li>
        <li><a href='https://www.youtube.com/watch?v=pOYAXv15r3A'>Forecasting at Scale: How and Why We Developed Prophet for Forecasting at Facebook</a></li>
    </ul>
</div>
