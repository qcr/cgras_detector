# Copyright 2024 - Andrew Kwok Fai LUI,
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

import os
import cv2
import pandas as pd
from base64 import b64encode
import plotly.graph_objects as go
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from cgras_datatools.logging_tools import logger
from detector.model import DETECT_DAO
from detector.models.detect import CoralObjectDetectModel
from detector.task_detection import DetectionTaskModel

WIDTH_OPTIONS = [
    {'label': '720p',  'value': 1280},
    {'label': '1080p', 'value': 1920},
]


class CountImagePreviewBlock():
    DEFAULT_SCALE_PCT = 5

    def __init__(self, app, prefix: str, container_id: str = None):
        self.app = app
        self.prefix = prefix + 'cip_'
        self._build_panel()
        self._register_internal_callbacks()

    def get_panel(self):
        return self._panel

    def register_trigger(self, trigger_id: str):
        p = self.prefix
        self.app.callback(
            [Output(p+'top_panel',        'style',    allow_duplicate=True),
             Output(p+'tile_id_store',    'data',     allow_duplicate=True),
             Output(p+'image_data_store', 'data',     allow_duplicate=True),
             Output(p+'graph_area',       'children', allow_duplicate=True),
             Output(p+'scale_slider',     'value',    allow_duplicate=True),
             Output(p+'regen_btn',        'disabled', allow_duplicate=True),
             Output(p+'reset_btn',        'disabled', allow_duplicate=True),
             Output(p+'zoom_label',       'children', allow_duplicate=True),
             Output(p+'sample_dropdown',  'options',  allow_duplicate=True),
             Output(p+'sample_dropdown',  'value',    allow_duplicate=True),
             Output(p+'zoom_slider',      'disabled', allow_duplicate=True),
             Output(p+'zoom_slider',      'value',    allow_duplicate=True)],
            [Input(trigger_id, 'data')],
            prevent_initial_call=True, allow_duplicate=True,
        )(self._cb_on_tile_change())

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_panel(self):
        p = self.prefix

        warning_alert = dbc.Alert([
            html.Strong('Performance notice: '),
            'Scales above 20% may take tens of seconds and require several GB of server RAM. '
            'At 100% the source image can reach 30,000 × 30,000 pixels (~2.7 GB uncompressed). '
            'Use high scales only when fine detail is required.',
        ], color='warning', className='py-2 small mb-2')

        scale_slider = dcc.Slider(
            id=p+'scale_slider', min=1, max=100, step=1,
            value=self.DEFAULT_SCALE_PCT,
            marks={1: '1%', 5: '5%', 10: '10%', 25: '25%',
                   50: '50%', 75: '75%', 100: '100%'},
            tooltip={'placement': 'bottom', 'always_visible': True},
            className='mt-1',
        )

        width_radio = dbc.RadioItems(
            id=p+'width_radio', options=WIDTH_OPTIONS,
            value=1280, inline=True, className='fs-6 mt-1',
        )

        sample_dropdown = dcc.Dropdown(
            id=p+'sample_dropdown',
            placeholder='No samples available',
            clearable=False,
            className='mb-2',
        )

        regen_btn = dbc.Button('Generate Image Preview',
                               id=p+'regen_btn', color='primary',
                               size='sm', disabled=True)
        reset_btn = dbc.Button('⟳ Reset to Fit',
                               id=p+'reset_btn', color='outline-secondary',
                               size='sm', className='ms-2', disabled=True)
        zoom_label = html.Span('—', id=p+'zoom_label',
                               className='ms-2 fw-bold text-primary fs-6 align-middle')
        zoom_slider = dcc.Slider(
            id=p+'zoom_slider', min=1, max=200, step=1,
            value=100,
            marks={1: '1%', 50: '50%', 100: '100%', 150: '150%', 200: '200%'},
            tooltip={'placement': 'bottom', 'always_visible': True},
            className='mt-1',
            disabled=True,
        )
        graph_area = dcc.Loading(
            id=p+'loading', type='default',
            children=html.Div(
                id=p+'graph_area',
                children=html.P('Select a tile to enable the preview.',
                                className='text-muted mt-3 text-center'),
                style={'overflowX': 'auto'},
            ),
        )

        self._panel = html.Div(id=p+'top_panel', children=[
            dcc.Store(id=p+'tile_id_store'),
            dcc.Store(id=p+'image_data_store'),
            # dummy output target for the clientside callbacks
            html.Div(id=p+'reset_dummy', style={'display': 'none'}),
            html.H4(dbc.Badge('HIGH-RESOLUTION IMAGE PREVIEW',
                              className='ms-1 me-2', color='white', text_color='secondary')),
            warning_alert,
            dbc.Row([
                dbc.Col([
                    html.Label('Preview scale (% of source resolution)',
                               className='fw-semibold mb-1'),
                    html.Small('Note: this sets the image resolution loaded from disk, '
                               'not the preview zoom level below.',
                               className='text-muted d-block mb-1'),
                    scale_slider,
                ], className='col-6'),
                dbc.Col([
                    html.Label('Display width', className='fw-semibold mb-1 d-block'),
                    width_radio,
                ], className='col-3'),
                dbc.Col([
                    html.Label('Preview zoom level', className='fw-semibold mb-1'),
                    dbc.Row([
                        dbc.Col(zoom_slider, className='col-8'),
                        dbc.Col(zoom_label, className='col-4 d-flex align-items-center'),
                    ], className='align-items-center g-0'),
                ], className='col-3'),
            ], className='mb-2 align-items-start'),
            dbc.Row([
                dbc.Col([
                    html.Label('Sample date', className='fw-semibold mb-1 d-block'),
                    sample_dropdown,
                ], className='col-12'),
            ], className='mb-2'),
            dbc.Row([
                dbc.Col([regen_btn, reset_btn],
                        className='col-auto d-flex align-items-center'),
            ], className='mb-2 mt-1'),
            graph_area,
        ], style={'visibility': 'hidden'})

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _register_internal_callbacks(self):
        p = self.prefix

        # Client-side "Reset to Fit": restores initial full-image axis ranges
        self.app.clientside_callback(
            f"""
            function(n_clicks, image_data) {{
                var no_update = window.dash_clientside.no_update;
                if (!n_clicks || !image_data) return ['', no_update, no_update];
                var gd = document.getElementById('{p}graph');
                if (gd) {{
                    Plotly.relayout(gd, {{
                        'xaxis.range[0]': 0, 'xaxis.range[1]': image_data.prev_w,
                        'yaxis.range[0]': 0, 'yaxis.range[1]': image_data.prev_h
                    }});
                }}
                var D = image_data.display_w || image_data.prev_w;
                var fit_zoom = Math.max(1, Math.min(200, Math.round(D / image_data.prev_w * 100)));
                return ['', fit_zoom + '%', fit_zoom];
            }}
            """,
            [Output(p+'reset_dummy', 'children'),
             Output(p+'zoom_label',  'children', allow_duplicate=True),
             Output(p+'zoom_slider', 'value',    allow_duplicate=True)],
            [Input(p+'reset_btn', 'n_clicks'),
             State(p+'image_data_store', 'data')],
            prevent_initial_call=True,
        )

        # Client-side: post-relayout range clamping + middle-mouse pan.
        # Plotly's native dragmode='pan' and scrollZoom handle the interactions;
        # we clamp after every relayout so the view never escapes [0,W]×[0,H].
        # No `applying` flag: Plotly fires plotly_relayout AFTER updating
        # gd._fullLayout, so a corrective relayout reads its own clamped result
        # on the next call and exits cleanly — no infinite loop.
        self.app.clientside_callback(
            f"""
            function(image_data) {{
                if (!image_data) return window.dash_clientside.no_update;

                // Remove stale global mouse listeners from the previous image.
                if (window._cip_mm_move) {{ window.removeEventListener('mousemove', window._cip_mm_move); window._cip_mm_move = null; }}
                if (window._cip_mm_up)   {{ window.removeEventListener('mouseup',   window._cip_mm_up);   window._cip_mm_up   = null; }}

                var W = image_data.prev_w, H = image_data.prev_h;
                var D = image_data.display_w || W;

                // Shift [n0,n1] inside [0,limit] without shrinking the window.
                function clamp1D(n0, n1, limit) {{
                    if (n1 - n0 > limit) {{ return [0, limit]; }}
                    if (n0 < 0)          {{ n1 -= n0; n0 = 0; }}
                    if (n1 > limit)      {{ n0 -= (n1 - limit); n1 = limit; }}
                    return [n0, n1];
                }}

                function attachToGraph(gd) {{
                    // Remove stale Plotly event handler and middle-mouse listener.
                    try {{ if (gd._cip_relayout_handler) gd.off('plotly_relayout', gd._cip_relayout_handler); }} catch(e) {{}}
                    if (gd._cip_mm_down) {{ try {{ gd.removeEventListener('mousedown', gd._cip_mm_down); }} catch(e) {{}} }}

                    // --- Post-relayout clamp (covers native pan, scroll zoom, modebar) ---
                    // Called after every plotly_relayout. Reads the already-committed
                    // range from gd._fullLayout and corrects only when out of bounds.
                    // Safe without a re-entrancy flag: if the corrective relayout fires
                    // plotly_relayout again, gd._fullLayout already holds the clamped
                    // value, so xOk && yOk is true and we return immediately.
                    var onRelayout = function() {{
                        if (!gd._fullLayout || !gd._fullLayout.xaxis) return;
                        var xr = gd._fullLayout.xaxis.range;
                        var yr = gd._fullLayout.yaxis.range;
                        if (!xr || !yr || xr.length < 2 || yr.length < 2) return;
                        var cx = clamp1D(xr[0], xr[1], W);
                        var cy = clamp1D(yr[0], yr[1], H);
                        var xOk = Math.abs(cx[0]-xr[0]) < 0.01 && Math.abs(cx[1]-xr[1]) < 0.01;
                        var yOk = Math.abs(cy[0]-yr[0]) < 0.01 && Math.abs(cy[1]-yr[1]) < 0.01;
                        if (!xOk || !yOk) {{
                            Plotly.relayout(gd, {{
                                'xaxis.range[0]': cx[0], 'xaxis.range[1]': cx[1],
                                'yaxis.range[0]': cy[0], 'yaxis.range[1]': cy[1],
                                'xaxis.autorange': false, 'yaxis.autorange': false
                            }});
                        }}
                    }};
                    gd.on('plotly_relayout', onRelayout);
                    gd._cip_relayout_handler = onRelayout;

                    // --- Middle-mouse-button pan (pre-clamped, no Plotly native handler) ---
                    var dragging = false, sx, sy;
                    var onDown = function(e) {{
                        if (e.button !== 1) return;
                        e.preventDefault();
                        dragging = true; sx = e.clientX; sy = e.clientY;
                    }};
                    var onMove = function(e) {{
                        if (!dragging || !gd._fullLayout) return;
                        var dx = e.clientX - sx, dy = e.clientY - sy;
                        if (dx === 0 && dy === 0) return;
                        sx = e.clientX; sy = e.clientY;
                        var lay = gd._fullLayout;
                        var xr = lay.xaxis.range, yr = lay.yaxis.range;
                        var pw = lay._size.w, ph = lay._size.h;
                        if (!pw || !ph) return;
                        var xd = -dx * (xr[1]-xr[0]) / pw;
                        var yd = -dy * (yr[1]-yr[0]) / ph;
                        var cx = clamp1D(xr[0]+xd, xr[1]+xd, W);
                        var cy = clamp1D(yr[0]+yd, yr[1]+yd, H);
                        Plotly.relayout(gd, {{
                            'xaxis.range[0]': cx[0], 'xaxis.range[1]': cx[1],
                            'yaxis.range[0]': cy[0], 'yaxis.range[1]': cy[1],
                            'xaxis.autorange': false, 'yaxis.autorange': false
                        }});
                    }};
                    var onUp = function(e) {{ if (e.button === 1) dragging = false; }};

                    gd.addEventListener('mousedown', onDown);
                    window.addEventListener('mousemove', onMove);
                    window.addEventListener('mouseup',   onUp);
                    gd._cip_mm_down     = onDown;
                    window._cip_mm_move = onMove;
                    window._cip_mm_up   = onUp;

                    // --- Zoom in / zoom out modebar buttons — 10% step ---
                    // Capture-phase handler fires before Plotly's bubble-phase handler
                    // on the same element; stopImmediatePropagation prevents Plotly's
                    // default zoom step (≈ 30%) from also running.
                    function applyZoomStep(step) {{
                        if (!gd._fullLayout || !gd._fullLayout.xaxis) return;
                        var xr = gd._fullLayout.xaxis.range;
                        var yr = gd._fullLayout.yaxis.range;
                        var current_zoom = Math.round(D * 100 / Math.max(xr[1] - xr[0], 1));
                        var new_zoom = Math.max(1, Math.min(200, current_zoom + step));
                        if (new_zoom === current_zoom) return;
                        var cx = (xr[0] + xr[1]) / 2;
                        var cy = (yr[0] + yr[1]) / 2;
                        var vw = D * 100 / new_zoom;
                        var vh = D * 100 / new_zoom;
                        var nx = clamp1D(cx - vw/2, cx + vw/2, W);
                        var ny = clamp1D(cy - vh/2, cy + vh/2, H);
                        Plotly.relayout(gd, {{
                            'xaxis.range[0]': nx[0], 'xaxis.range[1]': nx[1],
                            'yaxis.range[0]': ny[0], 'yaxis.range[1]': ny[1],
                            'xaxis.autorange': false, 'yaxis.autorange': false
                        }});
                    }}
                    var modeBtns = gd.querySelectorAll('.modebar-btn');
                    for (var i = 0; i < modeBtns.length; i++) {{
                        var btn = modeBtns[i];
                        var title = btn.getAttribute('data-title');
                        if (title === 'Zoom in' || title === 'Zoom out') {{
                            if (btn._cip_zoom_step_handler) {{
                                btn.removeEventListener('click', btn._cip_zoom_step_handler, true);
                            }}
                            (function(step) {{
                                btn._cip_zoom_step_handler = function(e) {{
                                    e.stopImmediatePropagation();
                                    applyZoomStep(step);
                                }};
                            }})(title === 'Zoom in' ? 10 : -10);
                            btn.addEventListener('click', btn._cip_zoom_step_handler, true);
                        }}
                    }}

                    // --- Zoom slider (pointerup on the rc-slider container) ---
                    // Using a DOM event listener rather than a Dash clientside callback
                    // avoids the ambiguity between user drags and programmatic value
                    // updates (from the relayoutData callback), which made the Dash
                    // callback approach unreliable.
                    var sliderEl = document.getElementById('{p}zoom_slider');
                    if (sliderEl) {{
                        if (sliderEl._cip_zoom_handler) {{
                            sliderEl.removeEventListener('pointerup', sliderEl._cip_zoom_handler);
                        }}
                        var applySliderZoom = function() {{
                            var handle = sliderEl.querySelector('.rc-slider-handle');
                            if (!handle) return;
                            var zoom_pct = parseInt(handle.getAttribute('aria-valuenow'), 10);
                            if (!zoom_pct || !gd._fullLayout || !gd._fullLayout.xaxis) return;
                            var xr = gd._fullLayout.xaxis.range;
                            var yr = gd._fullLayout.yaxis.range;
                            var current_zoom = Math.round(D * 100 / Math.max(xr[1] - xr[0], 1));
                            if (current_zoom === zoom_pct) return;
                            var cx = (xr[0] + xr[1]) / 2;
                            var cy = (yr[0] + yr[1]) / 2;
                            var vw = D * 100 / zoom_pct;
                            var vh = D * 100 / zoom_pct;
                            var nx = clamp1D(cx - vw/2, cx + vw/2, W);
                            var ny = clamp1D(cy - vh/2, cy + vh/2, H);
                            Plotly.relayout(gd, {{
                                'xaxis.range[0]': nx[0], 'xaxis.range[1]': nx[1],
                                'yaxis.range[0]': ny[0], 'yaxis.range[1]': ny[1],
                                'xaxis.autorange': false, 'yaxis.autorange': false
                            }});
                        }};
                        sliderEl.addEventListener('pointerup', applySliderZoom);
                        sliderEl._cip_zoom_handler = applySliderZoom;
                    }}
                }}

                // Poll until Plotly has fully initialised gd._fullLayout (async after React render).
                function tryAttach(attempts) {{
                    var gd = document.getElementById('{p}graph');
                    if (gd && gd._fullLayout && gd._fullLayout.xaxis && typeof gd.on === 'function') {{
                        attachToGraph(gd);
                    }} else if (attempts > 0) {{
                        setTimeout(function() {{ tryAttach(attempts - 1); }}, 150);
                    }}
                }}
                tryAttach(20);  // polls up to 20 × 150 ms = 3 s

                return window.dash_clientside.no_update;
            }}
            """,
            Output(p+'reset_dummy', 'children', allow_duplicate=True),
            Input(p+'image_data_store', 'data'),
            prevent_initial_call=True,
        )

        # Load image from disk and build interactive figure
        self.app.callback(
            [Output(p+'graph_area',       'children', allow_duplicate=True),
             Output(p+'image_data_store', 'data',     allow_duplicate=True),
             Output(p+'reset_btn',        'disabled', allow_duplicate=True),
             Output(p+'zoom_label',       'children', allow_duplicate=True),
             Output(p+'zoom_slider',      'disabled', allow_duplicate=True),
             Output(p+'zoom_slider',      'value',    allow_duplicate=True)],
            [Input(p+'regen_btn', 'n_clicks'),
             State(p+'tile_id_store',    'data'),
             State(p+'scale_slider',     'value'),
             State(p+'width_radio',      'value'),
             State(p+'sample_dropdown',  'value')],
            prevent_initial_call=True,
        )(self._cb_generate_preview())

        # Rebuild figure at new display width (no file re-read)
        self.app.callback(
            [Output(p+'graph_area',       'children', allow_duplicate=True),
             Output(p+'zoom_label',       'children', allow_duplicate=True),
             Output(p+'zoom_slider',      'value',    allow_duplicate=True),
             Output(p+'image_data_store', 'data',     allow_duplicate=True)],
            [Input(p+'width_radio',      'value'),
             State(p+'image_data_store', 'data')],
            prevent_initial_call=True,
        )(self._cb_update_width())

        # Client-side: sync zoom_label and zoom_slider from Plotly relayoutData.
        # Runs clientside so the slider tracks scroll-wheel zoom without a server round-trip.
        self.app.clientside_callback(
            f"""
            function(relayout_data, image_data) {{
                var no_update = window.dash_clientside.no_update;
                if (!relayout_data || !image_data) return [no_update, no_update];
                var D = image_data.display_w || image_data.prev_w;
                if ('xaxis.range[0]' in relayout_data && 'xaxis.range[1]' in relayout_data) {{
                    var visible_w = Math.max(
                        relayout_data['xaxis.range[1]'] - relayout_data['xaxis.range[0]'], 1);
                    var zoom_pct = Math.round(D / visible_w * 100);
                    return [zoom_pct + '%', zoom_pct];
                }}
                if (relayout_data['xaxis.autorange'] || relayout_data['autosize']) {{
                    var fit_zoom = Math.max(1, Math.min(200, Math.round(D / image_data.prev_w * 100)));
                    return [fit_zoom + '%', fit_zoom];
                }}
                return [no_update, no_update];
            }}
            """,
            [Output(p+'zoom_label',  'children', allow_duplicate=True),
             Output(p+'zoom_slider', 'value',    allow_duplicate=True)],
            [Input(p+'graph', 'relayoutData'),
             State(p+'image_data_store', 'data')],
            prevent_initial_call=True,
        )

    # ------------------------------------------------------------------
    # Callback implementations
    # ------------------------------------------------------------------
    def _cb_on_tile_change(self):
        def on_tile_change(tile_id):
            if tile_id is None:
                raise PreventUpdate
            placeholder = html.P(
                'Click "Regenerate Image Preview" to load the image at the selected scale.',
                className='text-muted mt-3 text-center')
            try:
                coral_trend = DETECT_DAO.get_coral_count_trend_as_df(tile_id)
                if coral_trend is None or len(coral_trend) == 0:
                    options, default_value = [], None
                else:
                    options = [
                        {'label': f'{pd.to_datetime(row["batch_time"]).strftime("%Y-%m-%d")} '
                                  f'({int(row["age"])} days)',
                         'value': row['tile_sample_id']}
                        for _, row in coral_trend.iterrows()
                    ]
                    default_value = coral_trend.iloc[-1]['tile_sample_id']
            except Exception:
                logger.exception(f'ImagePreview: failed to build sample list for tile {tile_id}')
                options, default_value = [], None
            # style={} clears 'visibility:hidden' — must always be returned so
            # the panel becomes visible even if dropdown population fails.
            return ({}, tile_id, None, placeholder,
                    self.DEFAULT_SCALE_PCT, False, True, '—',
                    options, default_value, True, 100)
        return on_tile_change

    def _cb_generate_preview(self):
        def generate_preview(n_clicks, tile_id, scale_pct, width_value, tile_sample_id):
            if not n_clicks or not tile_id or tile_sample_id is None:
                raise PreventUpdate
            img_path, error = self._find_image_path(tile_sample_id)
            if error:
                return (dbc.Alert([html.Strong('Image preview unavailable. '), error],
                                  color='warning', className='mt-2'),
                        None, True, '—', True, 100)
            try:
                img_b64, (prev_w, prev_h), (orig_w, orig_h) = \
                    self._load_preview(img_path, scale_pct)
            except Exception as exc:
                logger.exception(f'ImagePreview: failed for tile_sample {tile_sample_id}')
                return (dbc.Alert(f'Failed to generate preview: {exc}',
                                  color='danger', className='mt-2'),
                        None, True, '—', True, 100)
            image_data = {
                'b64': img_b64, 'prev_w': prev_w, 'prev_h': prev_h,
                'orig_w': orig_w, 'orig_h': orig_h,
                'img_path': img_path, 'scale_pct': scale_pct,
                'display_w': width_value,
            }
            fit_zoom = max(1, min(200, round(width_value / prev_w * 100)))
            content = self._build_graph_content(image_data, width_value)
            return content, image_data, False, f'{fit_zoom}%', False, fit_zoom
        return generate_preview

    def _cb_update_width(self):
        def update_width(width_value, image_data):
            if not image_data:
                raise PreventUpdate
            updated = {**image_data, 'display_w': width_value}
            fit_zoom = max(1, min(200, round(width_value / image_data['prev_w'] * 100)))
            return self._build_graph_content(updated, width_value), f'{fit_zoom}%', fit_zoom, updated
        return update_width

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_graph_content(self, image_data: dict, display_w: int):
        p = self.prefix
        b64     = image_data['b64']
        prev_w  = image_data['prev_w']
        prev_h  = image_data['prev_h']
        orig_w  = image_data['orig_w']
        orig_h  = image_data['orig_h']
        img_path   = image_data['img_path']
        scale_pct  = image_data.get('scale_pct', '?')

        display_h = display_w

        # go.layout.Image (background image on data axes) fills the full square
        # without injecting yaxis.scaleanchor, which go.Image does automatically
        # and which causes the plot area to shrink to the image's own aspect ratio.
        fig = go.Figure()
        fig.add_layout_image(dict(
            source=f'data:image/jpeg;base64,{b64}',
            xref='x', yref='y',
            x=0, y=prev_h,          # top-left anchor in data coords (y-axis is up)
            sizex=prev_w, sizey=prev_h,
            sizing='stretch',       # fill the full axis-defined region
            opacity=1.0,
            layer='below',
        ))
        fig.update_layout(
            width=display_w, height=display_h,
            margin=dict(l=0, r=0, t=0, b=0),
            dragmode='pan',
            plot_bgcolor='#111',    # dark surround for non-square source images
        )
        fig.update_xaxes(
            showticklabels=False, showgrid=False, zeroline=False,
            range=[0, prev_w], fixedrange=False,
        )
        fig.update_yaxes(
            showticklabels=False, showgrid=False, zeroline=False,
            range=[0, prev_h], fixedrange=False,
        )

        graph = dcc.Graph(
            id=p+'graph', figure=fig,
            style={'width': f'{display_w}px', 'height': f'{display_h}px'},
            config={
                'displayModeBar': True,
                'displaylogo': False,
                # Remove interaction buttons superseded by custom JS handlers.
                # Keep zoomIn2d / zoomOut2d (modebar); those fire plotly_relayout
                # which the safety-net clamp catches.
                'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'toImage'],
                'scrollZoom': True,
            },
        )
        caption = (
            f'Scale: {scale_pct}%  |  '
            f'Preview: {prev_w:,} × {prev_h:,} px  |  '
            f'Source: {orig_w:,} × {orig_h:,} px  |  '
            f'File: {os.path.basename(img_path)}'
        )
        return html.Div([
            html.P(caption, className='text-muted small mb-1'),
            html.Div(graph, style={'overflowX': 'auto'}),
        ])

    def _find_image_path(self, tile_sample_id: int):
        logdata_folder = DetectionTaskModel.get_cache_folder(tile_sample_id)
        if not logdata_folder or not os.path.isdir(logdata_folder):
            return None, 'Sample data folder not found on disk.'

        orig_path = os.path.join(
            logdata_folder,
            CoralObjectDetectModel.ANNOTATED_WHOLE_RECO_ORIGINAL_SCALE_IMAGE_FILENAME)
        work_path = os.path.join(
            logdata_folder,
            CoralObjectDetectModel.ANNOTATED_WHOLE_RECO_IMAGE_FILENAME)
        if os.path.isfile(orig_path):
            return orig_path, None
        if os.path.isfile(work_path):
            return work_path, None

        reco_yaml    = os.path.join(logdata_folder, 'reco_model.yaml')
        loctile_yaml = os.path.join(logdata_folder, 'loctile_model.yaml')
        cod_yaml     = os.path.join(logdata_folder, 'coral_object_detect_model.yaml')
        if not os.path.isfile(reco_yaml):
            return None, ('Tile reconstruction (RECO stage) did not complete. '
                          'Check for homography or feature-matching errors.')
        if not os.path.isfile(loctile_yaml):
            return None, ('Tile corner location (LOCTILE stage) did not complete. '
                          'The black corner markers may not have been sufficiently visible.')
        if not os.path.isfile(cod_yaml):
            return None, ('Coral detection (DETECT stage) did not produce output. '
                          'Verify a valid YOLO model is configured for this species and age range.')
        return None, 'The annotated image file was not generated by the pipeline.'

    def _load_preview(self, img_path: str, scale_pct: int):
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f'OpenCV could not read: {img_path}')
        orig_h, orig_w = img.shape[:2]
        scale = scale_pct / 100.0
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        return b64encode(buf.tobytes()).decode(), (new_w, new_h), (orig_w, orig_h)
