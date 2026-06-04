# Copyright 2024 - Andrew Kwok Fai LUI,
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import io, zipfile
from base64 import b64encode
from datetime import datetime
import pandas as pd
import plotly.express as px
import dash
from dash import html, dcc, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from cgras_datatools.logging_tools import logger
from detector.model import DETECT_DAO, CONFIG, SystemConfigNames, ClassHierarchyPresentation
from detector.dao_detect import SampleStatusNames
from detector.models.visualize import CoralObjectMapModel, CoralObjectMapModelHelper
from detector.models.heatmap_tools import HeatmapHelper

dash.register_page(__name__)


class BulkExporterPage():
    def __init__(self, app):
        self.app = app
        self.prefix = 'bulk_exp_'
        self._build_layout()
        self._register_callbacks()

    def layout(self):
        return self._layout

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        p = self.prefix

        species_dropdown = dcc.Dropdown(
            id=p+'species_filter', multi=True,
            placeholder='All species', clearable=True,
            style={'minWidth': '180px'})

        season_dropdown = dcc.Dropdown(
            id=p+'season_filter', multi=True,
            placeholder='All seasons', clearable=True,
            style={'minWidth': '180px'})

        tile_id_input = dcc.Input(
            id=p+'tile_id_filter', type='text',
            placeholder='Tile ID contains…', debounce=True,
            style={'width': '220px', 'height': '36px', 'padding': '4px 8px'})

        date_picker_start = dcc.DatePickerSingle(
            id=p+'date_start', placeholder='Start date',
            clearable=True, display_format='YYYY-MM-DD',
            style={'fontSize': '13px'})

        date_picker_end = dcc.DatePickerSingle(
            id=p+'date_end', placeholder='End date',
            clearable=True, display_format='YYYY-MM-DD',
            style={'fontSize': '13px'})

        filter_row = dbc.Row([
            dbc.Col([html.Label('Species',             className='fw-semibold mb-1'), species_dropdown],  className='col-auto'),
            dbc.Col([html.Label('Season',              className='fw-semibold mb-1'), season_dropdown],   className='col-auto'),
            dbc.Col([html.Label('Tile ID',             className='fw-semibold d-block mb-1'), tile_id_input],     className='col-auto'),
            dbc.Col([html.Label('Capture Start Date',  className='fw-semibold mb-1'), date_picker_start], className='col-auto'),
            dbc.Col([html.Label('Capture End Date',    className='fw-semibold mb-1'), date_picker_end],   className='col-auto'),
        ], className='g-3 align-items-end mb-3')

        columns = [
            {'name': 'Tile ID',         'id': 'tile_id',       'type': 'text'},
            {'name': 'Species',          'id': 'species',       'type': 'text'},
            {'name': 'Season',           'id': 'season',        'type': 'text'},
            {'name': 'Settled On',       'id': 'settle_time',   'type': 'text'},
            {'name': 'DONE Samples',     'id': 'done_count',    'type': 'numeric'},
            {'name': 'Latest Sample',    'id': 'latest_sample', 'type': 'text'},
        ]
        table = dash_table.DataTable(
            id=p+'datatable',
            columns=columns,
            data=[],
            row_selectable='multi',
            selected_rows=[],
            page_action='native',
            page_size=50,
            sort_action='native',
            filter_action='native',
            style_cell={'fontSize': 13, 'textAlign': 'left',
                        'padding': '6px 10px', 'whiteSpace': 'normal'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
            style_data_conditional=[
                {'if': {'state': 'selected'},
                 'backgroundColor': '#d4edda', 'border': '1px solid #28a745'},
            ],
            style_table={'overflowX': 'auto', 'borderLeft': '1px solid #dee2e6'},
        )

        action_bar = dbc.Row([
            dbc.Col([
                dbc.Button('Select All',   id=p+'select_all_btn',   color='secondary', size='sm', className='me-2'),
                dbc.Button('Deselect All', id=p+'deselect_all_btn', color='secondary', size='sm'),
                dbc.Badge('0 tiles selected', id=p+'selection_badge',
                          color='secondary', className='ms-3 fs-6 align-middle'),
            ], className='col-auto d-flex align-items-center'),
            dbc.Col([
                dbc.Button(
                    'Export Tile Sample Results',
                    id=p+'export_btn', color='success', size='md',
                    disabled=True, className='float-end'),
                dcc.Download(id=p+'download'),
            ], className='col-auto ms-auto'),
        ], className='align-items-center mb-3')

        self._layout = dbc.Container([
            dcc.Store(id=p+'main_panel'),
            dbc.Row(html.H3('Bulk Export — Tile Sample Results', className='mt-3 mb-2')),
            dbc.Row(html.P(
                'Filter tiles by species, season or tile ID, then select which to export. '
                'Each selected tile produces one Excel workbook inside the downloaded ZIP.',
                className='text-muted mb-2')),
            dbc.Card(dbc.CardBody(filter_row), className='mb-3 shadow-sm'),
            action_bar,
            html.Hr(className='mt-0 mb-3'),
            dbc.Row(dbc.Col(table)),
        ], fluid=True)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _register_callbacks(self):
        p = self.prefix

        # Populate filter dropdowns once on page load
        self.app.callback(
            [Output(p+'species_filter', 'options'),
             Output(p+'season_filter',  'options')],
            [Input(p+'main_panel', 'id')],
            prevent_initial_call='initial_duplicate'
        )(self._cb_populate_filters())

        # Refresh table when any filter changes
        self.app.callback(
            [Output(p+'datatable', 'data'),
             Output(p+'datatable', 'selected_rows')],
            [Input(p+'species_filter', 'value'),
             Input(p+'season_filter',  'value'),
             Input(p+'tile_id_filter', 'value'),
             Input(p+'date_start',     'date'),
             Input(p+'date_end',       'date')],
            prevent_initial_call='initial_duplicate'
        )(self._cb_update_table())

        # Select / Deselect All
        self.app.callback(
            Output(p+'datatable', 'selected_rows', allow_duplicate=True),
            [Input(p+'select_all_btn',   'n_clicks'),
             Input(p+'deselect_all_btn', 'n_clicks'),
             State(p+'datatable', 'data')],
            prevent_initial_call=True
        )(self._cb_select_all())

        # Badge + export button enable/disable
        self.app.callback(
            [Output(p+'selection_badge', 'children'),
             Output(p+'selection_badge', 'color'),
             Output(p+'export_btn', 'disabled')],
            [Input(p+'datatable', 'selected_rows')]
        )(self._cb_update_badge())

        # Export
        self.app.callback(
            Output(p+'download', 'data'),
            [Input(p+'export_btn', 'n_clicks'),
             State(p+'datatable', 'selected_rows'),
             State(p+'datatable', 'data')],
            prevent_initial_call=True
        )(self._cb_export())

    # ------------------------------------------------------------------
    # Callback implementations
    # ------------------------------------------------------------------
    def _cb_populate_filters(self):
        def populate(_):
            species = [{'label': s, 'value': s}
                       for s in DETECT_DAO.list_species_in_tile_sample()]
            seasons = [{'label': f'{s} Season', 'value': s}
                       for s in DETECT_DAO.list_seasons_in_tile_sample()]
            return species, seasons
        return populate

    def _cb_update_table(self):
        def update_table(species_list, season_list, tile_id_filter, date_start, date_end):
            df = self._query_tiles_summary(species_list, season_list, tile_id_filter, date_start, date_end)
            return df.to_dict('records'), []
        return update_table

    def _cb_select_all(self):
        def select_all(sel_clicks, desel_clicks, data):
            if not data:
                return []
            if ctx.triggered_id == self.prefix + 'select_all_btn':
                return list(range(len(data)))
            return []
        return select_all

    def _cb_update_badge(self):
        def update_badge(selected_rows):
            n = len(selected_rows) if selected_rows else 0
            label = f'{n} tile{"s" if n != 1 else ""} selected'
            colour = 'success' if n > 0 else 'secondary'
            return label, colour, (n == 0)
        return update_badge

    def _cb_export(self):
        def export(n_clicks, selected_rows, data):
            if not n_clicks or not selected_rows or not data:
                raise PreventUpdate
            tile_ids = [data[i]['tile_id'] for i in selected_rows]
            logger.info(f'BulkExporter: exporting {len(tile_ids)} tiles: {tile_ids}')
            try:
                zip_bytes = self._generate_bulk_zip(tile_ids)
            except Exception:
                logger.exception('BulkExporter: ZIP generation failed')
                raise PreventUpdate
            encoded = b64encode(zip_bytes).decode()
            fname = f'bulk_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
            return dict(content=encoded, filename=fname, base64=True)
        return export

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _query_tiles_summary(self, species_list, season_list, tile_id_filter,
                             date_start=None, date_end=None):
        """Return one row per tile_id with DONE sample count, filtered."""
        df = DETECT_DAO.query_processed_tile_samples(the_period=0)
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=[
                'tile_id', 'species', 'season', 'settle_time', 'done_count', 'latest_sample'])

        if species_list:
            df = df[df['species'].isin(species_list)]
        if season_list:
            df = df[df['season'].isin(season_list)]
        if tile_id_filter:
            df = df[df['tile_id'].str.contains(tile_id_filter, case=False, na=False)]

        # Date range filters on batch_time (capture date). batch_time is stored as
        # 'YYYY-MM-DD HH:MM:SS'; slicing to [:10] gives an ISO date string for comparison.
        if date_start:
            df = df[df['batch_time'].str[:10] >= date_start]
        if date_end:
            df = df[df['batch_time'].str[:10] <= date_end]

        done_df = df[df['status'] == SampleStatusNames.DONE.value]

        # Summarise per tile_id (keep only tiles with at least one DONE sample)
        done_counts = done_df.groupby('tile_id').size().reset_index(name='done_count')
        if len(done_counts) == 0:
            return pd.DataFrame(columns=[
                'tile_id', 'species', 'season', 'settle_time', 'done_count', 'latest_sample'])

        tile_attrs = df.groupby('tile_id').agg(
            species=('species', 'first'),
            season=('season', 'first'),
            settle_time=('settle_time', 'first'),
            latest_sample=('batch_time', 'max'),
        ).reset_index()

        summary = tile_attrs.merge(done_counts, on='tile_id')
        summary = summary.sort_values(['season', 'species', 'tile_id'])
        return summary.reset_index(drop=True)

    # ------------------------------------------------------------------
    # ZIP / Excel generation
    # ------------------------------------------------------------------
    def _generate_bulk_zip(self, tile_ids: list) -> bytes:
        sink = io.BytesIO()
        with zipfile.ZipFile(sink, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            for tile_id in tile_ids:
                try:
                    excel_bytes = self._generate_excel_for_tile(tile_id)
                    zf.writestr(f'{tile_id}/{tile_id}_data.xlsx', excel_bytes)
                    logger.info(f'BulkExporter: packed Excel for {tile_id}')
                except Exception:
                    logger.exception(f'BulkExporter: failed Excel for {tile_id}')
                for filename, img_bytes in self._generate_figures_for_tile(tile_id):
                    try:
                        zf.writestr(f'{tile_id}/{filename}', img_bytes)
                    except Exception:
                        logger.exception(f'BulkExporter: failed figure {filename} for {tile_id}')
        return sink.getvalue()

    def _generate_figures_for_tile(self, tile_id: str):
        """Yield (filename, png_bytes) for trend chart, heatmaps and scatter plots."""
        coral_trend_model = DETECT_DAO.get_coral_count_trend_as_df(tile_id)
        if coral_trend_model is None or len(coral_trend_model) == 0:
            return

        coral_trend_model['batch_time'] = (
            pd.to_datetime(coral_trend_model['batch_time']).dt.date)
        coral_trend_model['batch_time'] = coral_trend_model.apply(
            lambda row: f'{row["batch_time"]} ({row["age"]} days)', axis=1)

        # --- Trend chart ---
        if len(coral_trend_model) > 1:
            trend_fig = px.line(coral_trend_model, x='age', y='coral_alive_count')
            trend_fig.update_traces(line=dict(color='rgb(255, 0, 0)', width=4))
            trend_fig.update_xaxes(
                title='Age (days since settlement)', visible=True,
                showticklabels=True, showgrid=True, gridwidth=1, gridcolor='LightGrey',
                range=[0, coral_trend_model.iloc[-1]['age']])
            trend_fig.update_yaxes(
                title='Coral count', visible=True,
                showticklabels=True, showgrid=True, gridwidth=1, gridcolor='LightGrey')
            trend_fig.update_layout(plot_bgcolor='rgb(255, 255, 225)')
            yield (f'{tile_id}_trend_chart.png', trend_fig.to_image(format='png'))

        # --- Heatmaps (all samples, alive-coral class) ---
        filter_class = CoralObjectMapModelHelper.VISCLASS_CORAL['value']
        color_scale = CONFIG.get(SystemConfigNames.HEATMAP_COLOUR_SCALE, None)
        default_count_range = HeatmapHelper.DEFAULT_COUNT_RANGE
        max_count = 0
        heatmap_figs = []
        latest_index = len(coral_trend_model) - 1
        for index in range(latest_index, -1, -1):
            sample = coral_trend_model.iloc[index]
            tile_sample_id = sample['tile_sample_id']
            title = f'Captured on {sample["batch_time"]}'
            if index == latest_index:
                title += ' (Latest)'
            try:
                vt_model = CoralObjectMapModel(tile_sample_id)
                count_map, count_label_map = vt_model.compute_object_count_map(
                    filter_class, count_range=default_count_range)
                count_total = int(count_map.sum())
                fig = HeatmapHelper.generate_plotly_heatmap(
                    count_map, count_label_map,
                    title=f'{title} ({count_total})',
                    fig_size=(640, 600), count_range=default_count_range,
                    color_scale=color_scale)
                fig.update_layout(margin=dict(l=5, r=5, b=10, t=50, pad=0),
                                  title=dict(font=dict(size=18, weight='bold'), yref='paper'))
                max_count = max(max_count, int(count_map.max()))
                heatmap_figs.append(fig)
            except Exception:
                logger.exception(f'BulkExporter: heatmap failed for {tile_sample_id}')
        for fig in heatmap_figs:
            fig.update_layout(coloraxis=dict(cmax=max_count))
        for i, fig in enumerate(heatmap_figs):
            yield (f'{tile_id}_heatmap_{i}.png', fig.to_image(format='png'))

        # --- Scatter plot (latest sample) ---
        scatter_colour_map = {}
        for i, cat in enumerate(ClassHierarchyPresentation):
            scatter_colour_map[cat.name] = px.colors.qualitative.G10[i]
        latest_sample = coral_trend_model.iloc[latest_index]
        try:
            detected_df = DETECT_DAO.query_detected_objects(
                latest_sample['tile_sample_id'],
                present_classes=(
                    ClassHierarchyPresentation.ALIVE_CORAL.value,
                    ClassHierarchyPresentation.DEAD_CORAL.value,
                    ClassHierarchyPresentation.OTHER.value))
            detected_df['present_class'] = detected_df['present_class'].apply(
                lambda c: ClassHierarchyPresentation(c).name)
            scatter_fig = px.scatter(
                detected_df, x='centre_x', y='centre_y',
                color='present_class', width=480, height=520,
                title=f'Objects on the tile on {latest_sample["batch_time"]} (Latest)',
                color_discrete_map=scatter_colour_map)
            scatter_fig.update_layout(
                margin=dict(l=5, r=5, t=60, b=5),
                plot_bgcolor='rgba(64, 64, 64, 1)',
                title=dict(font=dict(size=16, weight='bold')),
                legend=dict(orientation='h', yanchor='bottom', y=-0.20, xanchor='left', x=0.00))
            scatter_fig.update_xaxes(title='X Location', showgrid=False, range=[0, 1])
            scatter_fig.update_yaxes(title='Y Location', showgrid=False, range=[1, 0])
            yield (f'{tile_id}_scatterplot_0.png', scatter_fig.to_image(format='png'))
        except Exception:
            logger.exception(f'BulkExporter: scatter plot failed for {tile_id}')

    def _generate_excel_for_tile(self, tile_id: str) -> bytes:
        """Generate Excel bytes for one tile_id — mirrors CountResultDownloadBlock logic."""
        with io.BytesIO() as output:
            writer = pd.ExcelWriter(output, engine='xlsxwriter')

            tile_info_df = DETECT_DAO.get_tile_info_from_tile_sample(tile_id, to_dataframe=True)
            tile_info_df.to_excel(writer, sheet_name='TileInfo', index=False)

            tile_sample_df = DETECT_DAO.list_tile_samples_of_tile_id(
                tile_id, status=SampleStatusNames.DONE.value)
            tile_sample_df.drop(
                columns=['spawn_time', 'status', 'priority', 'metadata'], inplace=True)
            tile_sample_df.to_excel(writer, sheet_name='TileSamples', index=False)

            coral_count_trend_df = DETECT_DAO.get_coral_count_trend_as_df(tile_id)
            coral_count_trend_df.to_excel(writer, sheet_name='CoralCountTrend', index=False)

            species     = tile_info_df.loc[0, 'species']
            map_size    = (tile_info_df.loc[0, 'tab_ncols'], tile_info_df.loc[0, 'tab_nrows'])
            tab_index_list = [(x, y) for y in range(map_size[1]) for x in range(map_size[0])]

            for _, row in coral_count_trend_df.iterrows():
                tile_sample_id = row['tile_sample_id']
                age            = row['age']
                batch_time     = row['batch_time']

                # skip samples not in the DONE tile_sample list
                if len(tile_sample_df[tile_sample_df['id'] == tile_sample_id]) == 0:
                    continue

                detected_df = DETECT_DAO.query_detected_objects(tile_sample_id)
                detected_df.drop(columns=['metadata'], inplace=True)
                detected_df.insert(1, 'tile_id',    tile_id)
                detected_df.insert(2, 'batch_time', batch_time)
                detected_df.insert(3, 'age',        age)
                detected_df.insert(4, 'species',    species)
                detected_df['tab_xindex'] = (
                    (detected_df['centre_x'] * map_size[0]).clip(upper=map_size[0]-1).astype(int))
                detected_df['tab_yindex'] = (
                    (detected_df['centre_y'] * map_size[1]).clip(upper=map_size[1]-1).astype(int))
                sheet_date = batch_time[:10]
                detected_df.to_excel(writer, sheet_name=f'Detect-{sheet_date}', index=False)

                # Tab-level count pivot
                pres = (detected_df
                        .groupby(['tab_xindex', 'tab_yindex', 'present_class'])
                        .size().reset_index()
                        .pivot(index=['tab_xindex', 'tab_yindex'], columns='present_class'))
                coral = (detected_df
                         .groupby(['tab_xindex', 'tab_yindex', 'coral_class'])
                         .size().reset_index()
                         .pivot(index=['tab_xindex', 'tab_yindex'], columns='coral_class'))
                merged = pd.merge(pres, coral, left_index=True, right_index=True)
                merged.columns = [c[1] for c in merged.columns.to_flat_index()]
                merged = (merged.reindex(tab_index_list)
                                .fillna(0).astype(int).reset_index())
                merged.insert(0, 'species',    species)
                merged.insert(1, 'tile_id',    tile_id)
                merged.insert(2, 'batch_time', batch_time)
                merged.insert(3, 'age',        age)
                merged.to_excel(writer, sheet_name=f'Detect-Tabs-{sheet_date}', index=False)

                # Count maps
                vt_model = CoralObjectMapModel(tile_sample_id)
                count_map, _ = vt_model.compute_object_count_map(
                    CoralObjectMapModelHelper.VISCLASS_CORAL['value'])
                pd.DataFrame(count_map).to_excel(
                    writer, sheet_name=f'CM-ALIVE-{sheet_date}', index=False)
                for coral_class in DETECT_DAO.list_coral_classes(tile_sample_id=tile_sample_id):
                    count_map, _ = vt_model.compute_object_count_map(coral_class)
                    pd.DataFrame(count_map).to_excel(
                        writer, sheet_name=f'CM-{coral_class}-{sheet_date}', index=False)

            writer.close()
            output.seek(0)
            return output.getvalue()
