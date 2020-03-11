import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr


sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

filepath = '/Users/kenzatazi/Downloads/head_of_soils_recommendations_MGM.csv'

filepath1 = '/Users/kenzatazi/Downloads/FAOSTAT_data_3-11-2020.csv'
filepath2 = '/Users/kenzatazi/Downloads/Data_Extract_From_WDI_Database_Archives_(beta)/291aee82-3d52-4642-abb2-337c201bfa47_Data.csv'


def historical_yield(filepath):
    """ Returns the bar chart with the ten largest producers of maize """
    

    plt.figure()


def agroclimatic_indicators(filepath):
    """ returns facet grid plot of the different indicators for 2010"""
    # Data (look at spread in different places over one year)

    df_raw =  pd.read_csv(filepath)
    
    # Frost days (days)
    FD = pd.DataFrame({'FD': (df_raw[['FD-01-05-2010', 'FD-02-05-2010', 'FD-03-05-2010', 'FD-04-05-2010',
                                      'FD-05-05-2010', 'FD-06-05-2010', 'FD-07-05-2010', 'FD-08-05-2010', 
                                      'FD-09-05-2010', 'FD-10-05-2010', 'FD-11-05-2010', 'FD-12-05-2010', 
                                      'FD-01-15-2010', 'FD-02-15-2010', 'FD-03-15-2010', 'FD-04-15-2010',
                                      'FD-05-15-2010', 'FD-06-15-2010', 'FD-07-15-2010', 'FD-08-15-2010', 
                                      'FD-09-15-2010', 'FD-10-15-2010']].values).flatten()})
    FD = FD.melt(value_vars='FD')

    # Biological Effective Degree Days (°C)
    BEDD =  pd.DataFrame({'BEDD': (df_raw[['BEDD-01-05-2010', 'BEDD-02-05-2010', 'BEDD-03-05-2010',
                                           'BEDD-04-05-2010', 'BEDD-05-05-2010', 'BEDD-06-05-2010',
                                           'BEDD-07-05-2010', 'BEDD-08-05-2010', 'BEDD-09-05-2010',
                                           'BEDD-10-05-2010', 'BEDD-11-05-2010', 'BEDD-12-05-2010',
                                           'BEDD-01-15-2010', 'BEDD-02-15-2010', 'BEDD-03-15-2010',
                                           'BEDD-04-15-2010', 'BEDD-05-15-2010', 'BEDD-06-15-2010',
                                           'BEDD-07-15-2010', 'BEDD-08-15-2010', 'BEDD-09-15-2010',
                                           'BEDD-10-15-2010', 'BEDD-11-15-2010', 'BEDD-12-15-2010', 
                                           'BEDD-01-25-2010', 'BEDD-02-25-2010', 'BEDD-03-25-2010',
                                           'BEDD-04-25-2010', 'BEDD-05-25-2010', 'BEDD-06-25-2010',
                                           'BEDD-07-25-2010', 'BEDD-08-25-2010', 'BEDD-09-25-2010', 
                                           'BEDD-10-25-2010', 'BEDD-11-25-2010', 'BEDD-12-25-2010']].values).flatten()})
    BEDD = BEDD.melt(value_vars='BEDD')

    # Warm Spell Duration Index (days)
    WSDI = pd.DataFrame({'WSDI': (df_raw[['WSDI-Q1-2010', 'WSDI-Q2-2010',
                         'WSDI-Q3-2010', 'WSDI-Q4-2010']].values).flatten()})
    WSDI = WSDI.melt(value_vars='WSDI') 

    # Cold Spell Duration Index (days)
    CSDI = pd.DataFrame({'CSDI': (df_raw[['CSDI-Q1-2010', 'CSDI-Q2-2010', 
                        'CSDI-Q3-2010', 'CSDI-Q4-2010']].values).flatten()})
    CSDI = CSDI.melt(value_vars='CSDI')                    
   
    # Warm and Wet days (days)
    WW = pd.DataFrame({'WW':(df_raw[['WW-Q1-2010', 'WW-Q2-2010', 'WW-Q3-2010',
                       'WW-Q4-2010']].values).flatten()})
    WW = WW.melt(value_vars='WW')

    # Cumulative Wet Days (days)
    CWD = pd.DataFrame({'CWD': (df_raw[['CWD-Q1-2010', 'CWD-Q2-2010', 'CWD-Q3-2010', 
                        'CWD-Q4-2010']].values).flatten()})
    CWD = CWD.melt(value_vars='CWD')

    # Cumulative Frost Days (days)
    CFD = pd.DataFrame({'CFD': (df_raw[['CFD-Q1-2010', 'CFD-Q2-2010', 'CFD-Q3-2010',
                        'CFD-Q4-2010']].values).flatten()})
    CFD = CFD.melt(value_vars='CFD')

    # Cumulative Dry Days (days)
    CDD = pd.DataFrame({'CDD': (df_raw[['CDD-Q1-2010', 'CDD-Q2-2010', 'CDD-Q3-2010',
                        'CDD-Q4-2010']].values).flatten()})
    CDD = CDD.melt(value_vars='CDD')

    df = FD.append([WW], ignore_index=True)

    # FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="variable", hue="variable", aspect=15, height=.5, palette=pal)

    # Densities 
    g.map(sns.kdeplot, "value", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    # g.map(sns.kdeplot, "value", clip_on=False, color="w", lw=2, bw=.2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "value")

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
 

def soil_types(filepath):
    df_raw =  pd.read_csv(filepath)
    df = df_raw['soil_types']
    plt.figure()
    plt.title('Soil quality')
    plt.ylabel('Count')
    plt.xlabel('Soil type')
    df.hist(bins=107, align='mid', rwidth=0.75)
    plt.show()


def gdp(filepath1, filepath2):
    yield_df = pd.read_csv(filepath1)
    gdp_df = pd.read_csv(filepath2)
    gdp_df['GDP'] = pd.to_numeric(gdp_df['2010 [YR2010]'], errors='coerce')
    

    big_df = pd.concat([yield_df, gdp_df], axis=1, sort=False)
    df = big_df.dropna()
    
    
    df.plot.scatter(x='GDP', y='Value')
    plt.title('Yield as a function of GDP per capita')
    plt.xlabel('GDP per capita (USD)')
    plt.ylabel('Yield (tonnes/hectare)')


def climate_zones(filepath):

    """ bar chart with climate zones of haversted areas """

    names = ['Tropics', 'Subtropics \n (summer rainfall)',
             'Subtropics\n(winter\nrainfall)', 'Temperate\n(oceanic)',
             'Temperate\n(sub-\ncontinental)', 'Temperate\n(continental)',
             'Boreal\n(oceanic)', 'Boreal\n(sub-\ncontinental)',
             'Boreal \n(continental)', 'Arctic']
    
    df_raw =  pd.read_csv(filepath)
    df = df_raw['climate_zones']
    df_clean = df[df >= 0]

    hist, bin_edges = np.histogram(df_clean)
    
    plt.figure()
    plt.title('Climate zones')
    plt.bar(names, float(hist)/len(df_clean))
    plt.show()
