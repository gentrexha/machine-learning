import sys
from src.utils import *
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.io as pio


def analyze_crime_dataset():
    df = load_raw_data('crime')
    # group Violent crime and nonViolent crime by state
    crimedata_state = df.groupby('state').agg({'ViolentCrimesPerPop': 'mean', 'nonViolPerPop': 'mean'})[
        ['ViolentCrimesPerPop', 'nonViolPerPop']].reset_index()

    ##Aggregate view of Non-Violent Crimes by State
    data1 = dict(type='choropleth',
                 colorscale=[[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'],
                             [0.2222222222222222, 'rgb(244,109,67)'], [0.3333333333333333, 'rgb(253,174,97)'],
                             [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'],
                             [0.6666666666666666, 'rgb(171,217,233)'], [0.7777777777777778, 'rgb(116,173,209)'],
                             [0.8888888888888888, 'rgb(69,117,180)'], [1.0, 'rgb(49,54,149)']].reverse(),
                 autocolorscale=False,
                 locations=crimedata_state['state'],
                 locationmode='USA-states',
                 z=crimedata_state['nonViolPerPop'].astype(float),
                 colorbar={'title': 'non-Violent Crimes(Per-100K-Pop)'}
                 )
    layout1 = dict(
        title='Aggregate view of non-Violent Crimes Per 100K Population',
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showlakes=False,
            lakecolor='rgb(85,173,240)'),
    )

    fig1 = go.Figure(data=[data1], layout=layout1)
    # pio.write_image(fig1, REPORTS['crime']['figures'] / '{}{}'.format('crime', '_non_violent_crime_map.png'))
    plot(fig1, validate=False)

    # Aggregate view of Violent Crimes by State
    data2 = dict(type='choropleth',
                 autocolorscale=False,
                 colorscale=[[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'],
                             [0.2222222222222222, 'rgb(244,109,67)'], [0.3333333333333333, 'rgb(253,174,97)'],
                             [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'],
                             [0.6666666666666666, 'rgb(171,217,233)'], [0.7777777777777778, 'rgb(116,173,209)'],
                             [0.8888888888888888, 'rgb(69,117,180)'], [1.0, 'rgb(49,54,149)']].reverse(),
                 locations=crimedata_state['state'],
                 locationmode='USA-states',
                 z=crimedata_state['ViolentCrimesPerPop'].astype('float'),
                 colorbar={'title': 'Violent Crimes(Per-100K-Pop)'}
                 )
    layout2 = dict(
        title='Aggregate view of Violent Crimes Per 100K Population across US',
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showlakes=False,
            lakecolor='rgb(85,173,240)'),
    )

    fig2 = go.Figure(data=[data2], layout=layout2)
    # pio.write_image(fig2, REPORTS['crime']['figures'] / '{}{}'.format('crime', '_violent_crime_map.png'))
    plot(fig2, validate=False)

    # Boxplot of Violent crime variables
    plt.figure()
    df.boxplot(column=['murdPerPop', 'rapesPerPop', 'robbbPerPop', 'assaultPerPop', 'ViolentCrimesPerPop'])
    plt.savefig(REPORTS['crime']['figures'] / '{}{}'.format('crime', '_violent_crime_boxplot.png'), dpi=300)
    plt.clf()


def main():
    # analyze_raw_dataset('beijing_pm25')
    # analyze_raw_dataset('bike_sharing')
    # analyze_raw_dataset('crime')
    # analyze_raw_dataset('student_performance')
    analyze_crime_dataset()


if __name__ == '__main__':
    main()
    sys.exit(0)
