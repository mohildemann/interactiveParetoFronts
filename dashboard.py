import numpy as np
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import pickle
import os.path
from tifffile import tifffile
import json
from pyproj import Proj, transform
import pandas as pd
import dash
import dash_bootstrap_components as dbc
import imagecodecs
inProj = Proj('epsg:32637')
outProj = Proj('epsg:4326')

# Methods for creating components in the layout code
def Card(children, **kwargs):
    return html.Section(children, className="card-style")


def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        # style={"margin": "25px 5px 30px 0px"},
        children=[
            f"{name}:",
            html.Div(
                # style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )


def NamedInlineRadioItems(name, short, options, val, **kwargs):
    return html.Div(
        id=f"div-{short}",
        # style={"display": "inline-block"},
        children=[
            f"{name}:",
            dcc.RadioItems(
                id=f"radio-{short}",
                options=options,
                value=val,
                labelStyle={"display": "inline-block", "margin-right": "7px"},
                # style={"display": "inline-block", "margin-left": "7px"},
            ),
        ],
    )

def create_layout(app):
    # Actual layout of the app
    return html.Div(

        className="row",
        style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 10px", "margin": "10px 0px"},
        children=[
            dbc.Row(
                [

                    dbc.Col(

                        # Header
                        html.Div(

                            html.H3(
                                "Landuse optimization",
                                className="header_title",
                                id="app-title",
                            ))
                    ),
                ],
                justify="start"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            dcc.Dropdown(
                                id="dropdown-objectives",
                                searchable=False,
                                clearable=False,
                                options=[
                                    {
                                        "label": "Water Yield - Crop Yield",
                                        "value": '[3,2]',
                                    },
                                    {
                                        "label": "Habitat Heterogenerity - Crop Yield",
                                        "value": '[0,2]',
                                    },
                                    {
                                        "label": "Species Richness - Crop Yield",
                                        "value": '[1,2]',
                                    },
                                ],
                                placeholder="Select the pairwise objectives",
                                value='[3,2]',
                            )), width={"size": "auto", "offset": 1}, lg=3), ]),
            dbc.Row(
                [
                    dbc.Col(html.Div(
                        className="six columns",
                        children=[
                            dcc.Graph(id="graph-3d-plot-tsne"
                                      # , style={"width": 6}
                                      )
                        ],
                    ), width=9),
                    dbc.Col(html.Div(
                        className="three columns",
                        id="div-plot-click-wordemb",

                        # html.Div(id="div-plot-click-wordemb"),

                    )
                        , width=3
                    )

                ],

                # no_gutters=True,
                justify="center",
            ),

        ],
    )


def demo_callbacks(app, pareto_front, outputdirectory):
    def generate_figure_image(figure, points, layout, ):
        figure.add_trace(go.Scatter(
            x=points.iloc[:, 0],
            y=points.iloc[:, 1],

            # z=val["z"],
            showlegend=True,
            legendgroup="scatterpoints",
            textposition="top left",
            mode="markers",
            marker=dict(size=3, symbol="circle"),

        ))
        figure.update_layout(
            title={
                'y': 0.975,
                'x': 0.055,
                'xanchor': 'left',
                'yanchor': 'top'},
            legend={'itemsizing': 'constant'})
        # figure.update_layout(legend_title=labels[np.where(run_folders == run)][0])
        return figure

    @app.callback(
        Output("graph-3d-plot-tsne", "figure"),
        [Input("dropdown-objectives", "value"), ]
    )
    def display_3d_scatter_plot(objectives):

        def textToList(hashtags):
            return hashtags.strip('[]').replace('\'', '').replace(' ', '').split(',')

        axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)
        units = [" (HR Index)", " (SR Index)", " (CY Index)", " (WY Index)"]

        layout = go.Layout(
            title={'text': "Pairwise Pareto front of objectives {} and {}".format("Yearly soil loss",
                                                                                  "Required labour requirements"),
                   'x': 0.2, },
            margin=dict(l=20, r=20, b=20, t=40),
            scene=dict(xaxis=axes, yaxis=axes),
            width=1000, height=600,
            autosize=False,
            xaxis_title="Yearly soil loss in t/ha",
            yaxis_title="Labour requirements in labour days/ha",

        )
        figure = go.Figure(layout=layout)
        i = 0
        obj1_values = []
        obj2_values = []
        solution_ids = []
        for solution in pareto_front:
            for realization_id in range(len(solution.objective_values[0])):
                obj1_values.append(solution.objective_values[0][realization_id])
                obj2_values.append(solution.objective_values[1][realization_id])
                solution_ids.append(i)
            i += 1
        scattered_points = pd.DataFrame(
            {'obj1': np.array(obj1_values), 'obj2': np.array(obj2_values), 'sol_id': np.array(solution_ids)})
        figure = generate_figure_image(figure, scattered_points, layout)
        names = set()
        figure.for_each_trace(
            lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))
        return figure

    @app.callback(
        Output("div-plot-click-wordemb", "children"),
        [
            Input("graph-3d-plot-tsne", "clickData"), ],
    )
    def display_click_image(
            clickData
    ):
        if clickData:
            clicked_solution = None

            def textToList(hashtags):
                return hashtags.strip('[]').replace('\'', '').replace(' ', '').split(',')

            click_point_np = [float(clickData["points"][0][i]) for i in ["x", "y"]]

            filtered_solutions = []
            for solution in pareto_front:
                filtered_solutions.append(solution)
                if click_point_np[0] in solution.objective_values[0] and \
                        click_point_np[1] in solution.objective_values[1]:
                    clicked_solution = solution
                    patchmap_of_picked_solution = None

            try:
                if clickData and clicked_solution is not None:
                    img = tifffile.imread(
                        os.path.join(outputdirectory, str(clicked_solution._solution_id), 'rusle.tif'))
                    trace1 = go.Heatmap(
                        z=img)

                    layout1 = go.Layout(
                        title=f'Corresponding land use Map',
                        yaxis=dict(showticklabels=False),
                        xaxis=dict(showticklabels=False)
                    )

                    ws_lons = []
                    ws_lats = []
                    cl_lons = []
                    cl_lats = []

                    with open(os.path.join(outputdirectory, str(clicked_solution._solution_id),
                                           'protected_watersheds.geojson')) as json_file:
                        watersheds = json.load(json_file)
                    for feature in watersheds["features"]:
                        ws_feature_lats = np.array(feature["geometry"]["coordinates"])[:, :, 0].tolist()
                        ws_feature_lons = np.array(feature["geometry"]["coordinates"])[:, :, 1].tolist()
                        ws_lons = ws_lons + ws_feature_lons[0] + [None]
                        ws_lats = ws_lats + ws_feature_lats[0] + [None]

                    with open(os.path.join(outputdirectory, str(clicked_solution._solution_id),
                                           'terraces.geojson')) as json_file:
                        terraces = json.load(json_file)

                    for feature in terraces["features"]:
                        cl_feature_lats = np.array(feature["geometry"]["coordinates"])[:, 0].tolist()
                        cl_feature_lons = np.array(feature["geometry"]["coordinates"])[:, 1].tolist()
                        cl_lons = cl_lons + cl_feature_lons + [None]
                        cl_lats = cl_lats + cl_feature_lats + [None]

                    ws_lats = np.array(ws_lats)
                    ws_lons = np.array(ws_lons)
                    ws_not_None_ids = ws_lons != np.array(None)
                    ws_lats[ws_not_None_ids], ws_lons[ws_not_None_ids] = transform(inProj, outProj,
                                                                                   ws_lats[ws_not_None_ids],
                                                                                   ws_lons[ws_not_None_ids])

                    cl_lats = np.array(cl_lats)
                    cl_lons = np.array(cl_lons)
                    cl_not_None_ids = cl_lons != np.array(None)
                    cl_lats[cl_not_None_ids], cl_lons[cl_not_None_ids] = transform(inProj, outProj,
                                                                                   cl_lats[cl_not_None_ids],
                                                                                   cl_lons[cl_not_None_ids])
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scattermapbox(
                            lat=ws_lats,
                            lon=ws_lons,
                            mode="lines",
                            fill="toself",
                            line=dict(width=1, color='aliceblue'),

                        )
                    )

                    fig.add_trace(
                        go.Scattermapbox(
                            lat=cl_lats,
                            lon=cl_lons,
                            mode="lines",
                            line=dict(width=0.5, color="#F00"),

                        )
                    )

                    fig.update_layout(
                        margin={"r": 0, "t": 0, "l": 0, "b": 0},
                        mapbox=go.layout.Mapbox(
                            style="stamen-terrain",
                            zoom=11,
                            center_lat=10.4,
                            center_lon=37.73,
                        )
                    )
                    # fig.add_trace(trace1)

                    # fig.add_trace(
                    #     go.Scattermapbox(
                    #         lat=np.array([feature["geometry"]["coordinates"])[:, 1],
                    #         lon=np.array(feature["geometry"]["coordinates"])[:, 0],
                    #         mode="lines",
                    #         line=dict(width=8, color="#F00")
                    #     )
                    # )
                    #
                    # fig.update_layout(
                    #     title_text='Feb. 2011 American Airline flight paths<br>(Hover for airport names)',
                    #     showlegend=False,
                    #     geo=go.layout.Geo(
                    #         scope='north america',
                    #         projection_type='azimuthal equal area',
                    #         showland=True,
                    #         landcolor='rgb(243, 243, 243)',
                    #         countrycolor='rgb(204, 204, 204)',
                    #     ),
                    #     height=700,
                    # )

                    # fig.add_trace(trace2)
                    # fig.update_layout(height=800, width=500)
                    # fig.show()
                    # fig.update_yaxes(showticklabels=False, row=1, col=1)
                    # fig.update_yaxes(showticklabels=False, row=2, col=1)
                    # fig.update_xaxes(showticklabels=False, row=1, col=1)
                    # fig.update_xaxes(showticklabels=False, row=2, col=1)
                    return dcc.Graph(
                        id="graph-bar-nearest-neighbors-word",
                        figure=fig,
                    )
            except  KeyError as error:
                raise PreventUpdate
    return None

class Solution:
    _id = 0

    def __init__(self, representation, objective_values):
        self._solution_id = Solution._id
        Solution._id += 1
        self.representation = representation
        self.objective_values = objective_values
        # self.metadata = metadata

with open(os.path.join("input", 'all_populations5.pkl'), 'rb') as handle:
    populations = pickle.load(handle)

gis_dir= "input/gis_data"
final_population = populations[-1]
final_population_objective_values = [F for F in final_population[0]]

# correction to soil loss per ha, total size of watershed polygons is 6872.8406 ha
final_population_objective_values = [[F[0] / 6872.8406, F[1] / 6872.8406] for F in final_population_objective_values]
final_population_genes = [X for X in final_population[1]]
# final_population_metadata = [elem.data for elem in final_population_df[0]]
optimal_solutions = []
for i in range(len(final_population_objective_values)):
    optimal_solutions.append(Solution(final_population_genes[i], final_population_objective_values[i]))

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
app.layout = create_layout(app)
demo_callbacks(app, optimal_solutions, gis_dir)
app.run_server(debug=True)
