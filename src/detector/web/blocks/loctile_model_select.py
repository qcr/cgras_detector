import os, re, base64
import rospy
import numpy as np
import cv2
import dash
from dash import html, dcc, Input, Output, State, dash_table, ctx, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from cgras_datatools.logging_tools import logger
from detector.model import CONFIG
from detector.models.models_config import ModelsConfigNames

# Tile filter models live alongside locate_tile.py
_MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../models/tile_filter')
_CONFIG_KEY = ModelsConfigNames.LOCTILE_PIXEL_CLASSIFIER_FILENAME.value

_PARAM_KEYS = {
    'working_scale': ModelsConfigNames.LOCTILE_WORKING_SCALE.value,
    'template_size': ModelsConfigNames.LOCTILE_TEMPLATE_SIZE.value,
    'corner_size':   ModelsConfigNames.LOCTILE_TEMPLATE_CORNER_SIZE.value,
    'template_type': ModelsConfigNames.LOCTILE_TEMPLATE_TYPE.value,
}
_PARAM_DEFAULTS = {'working_scale': 0.1, 'template_size': 120, 'corner_size': 20, 'template_type': 'edge'}


# ---------------------------------------------------------------------------
# Template preview helpers
# ---------------------------------------------------------------------------

def _make_corner_preview(corner: str, ttype: str, size: int = 80) -> str:
    """Return a base64 PNG data URI of a corner template image."""
    img = np.full((size, size), 255, dtype=np.uint8)
    if ttype == 'edge':
        cs = max(1, size // 5)
        if corner == 'TL':
            img[cs:, cs:] = 0
        elif corner == 'TR':
            img[cs:, :size - cs] = 0
        elif corner == 'BR':
            img[:size - cs, :size - cs] = 0
        elif corner == 'BL':
            img[:size - cs, cs:] = 0
    else:  # block — all black, one quadrant white per _generate_template_block
        h = size // 2
        img[:] = 0
        if corner == 'TL':
            img[h:, h:] = 255  # bottom-right white
        elif corner == 'TR':
            img[h:, :h] = 255  # top-right white
        elif corner == 'BR':
            img[:h, :h] = 255  # top-left white
        elif corner == 'BL':
            img[:h, h:] = 255  # bottom-left white
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), 160, 1)
    _, buf = cv2.imencode('.png', img)
    return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode()


def _template_preview_card(ttype: str, title: str, note: str, badge_color: str) -> dbc.Card:
    corners = [('TL', 'Top-Left'), ('TR', 'Top-Right'), ('BL', 'Bottom-Left'), ('BR', 'Bottom-Right')]
    previews = []
    for cname, clabel in corners:
        src = _make_corner_preview(cname, ttype)
        previews.append(dbc.Col([
            html.Img(src=src, style={
                'width': '72px', 'height': '72px',
                'image-rendering': 'pixelated',
                'display': 'block', 'margin': '0 auto',
            }),
            html.Div(clabel, style={'font-size': '11px', 'text-align': 'center', 'marginTop': '2px'}),
        ], width=6, className='mb-2'))
    return dbc.Card([
        dbc.CardHeader(html.Div([
            dbc.Badge(title, color=badge_color, className='me-2'),
            html.Small(note, className='text-muted'),
        ])),
        dbc.CardBody(dbc.Row(previews)),
    ], className='h-100')


# ---------------------------------------------------------------------------
# Config persistence helpers
# ---------------------------------------------------------------------------

def _list_available_models():
    """Return a list of dicts for each .model file found in the tile_filter directory."""
    if not os.path.isdir(_MODELS_DIR):
        return []
    active = CONFIG.get(_CONFIG_KEY, '')
    rows = []
    for fname in sorted(os.listdir(_MODELS_DIR)):
        if not fname.endswith('.model'):
            continue
        exists = os.path.isfile(os.path.join(_MODELS_DIR, fname))
        rows.append({
            'Filename': fname,
            'Exists': 'Yes' if exists else 'No',
            'Active': 'Yes' if fname == active else 'No',
        })
    return rows


def _write_config_values(updates: dict) -> bool:
    """Update one or more keys in the runtime config yaml.

    For each key, regex-substitutes the existing line. If the key is absent it
    is inserted after the last existing loctile_ parameter line. Comments on
    unrelated lines are preserved.
    Updates CONFIG in-memory and the ROS parameter server for each key.
    """
    config_file = CONFIG.config_file
    try:
        with open(config_file, 'r') as f:
            content = f.read()

        for key, value in updates.items():
            new_content, n_subs = re.subn(
                rf'({re.escape(key)}:\s*)\S[^\n]*',
                rf'\g<1>{value}',
                content,
            )
            if n_subs == 0:
                # Key not present — insert after the last loctile_ line
                loctile_spans = list(re.finditer(r'  loctile_[^\n]+', content))
                if loctile_spans:
                    insert_at = loctile_spans[-1].end()
                    new_content = content[:insert_at] + f'\n  {key}: {value}' + content[insert_at:]
                else:
                    logger.warning(f'LocTileModelSelectBlock: no loctile section found; appending {key}')
                    new_content = content + f'  {key}: {value}\n'
            content = new_content

            CONFIG[key] = value
            try:
                rospy.set_param(f'~{key}', value)
            except Exception:
                logger.warning(f'LocTileModelSelectBlock: could not set ROS param ~{key}')

        with open(config_file, 'w') as f:
            f.write(content)
        return True
    except Exception:
        logger.exception('LocTileModelSelectBlock: failed to write config values')
        return False


def _save_active_model(filename: str) -> bool:
    return _write_config_values({_CONFIG_KEY: filename})


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class LocTileModelSelectBlock():
    def __init__(self, app, prefix):
        self.app = app
        self.prefix = prefix = prefix + 'ltms_'

        # --- shared toast ---
        self._toast = dbc.Toast(
            id=prefix + 'toast', is_open=False, duration=5000, icon='danger', header='Message',
            style={'position': 'fixed', 'top': '15%', 'left': '50%', 'width': 640,
                   'transform': 'translate(-50%, -50%)'},
        )

        # --- model selector: delete confirmation modal (kept for future use) ---
        self._confirm_modal = dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(id=prefix + 'confirm_modal_title')),
            html.Div([
                html.P(id=prefix + 'confirm_modal_message'),
                dbc.Button('Confirm', id={'type': prefix + 'action', 'index': 'confirm'}),
                dbc.Button('Cancel', id={'type': prefix + 'action', 'index': 'cancel'},
                           color='secondary'),
            ], className='d-grid gap-2 col-8 mx-auto', style={'padding': '6px'}),
        ], id=prefix + 'confirm_modal', is_open=False)

        # --- model selector: datatable ---
        self._datatable = dash_table.DataTable(
            data=[],
            columns=[{'name': c, 'id': c, 'type': 'text', 'editable': False}
                     for c in ('Filename', 'Exists', 'Active')],
            id=prefix + 'datatable',
            fill_width=True,
            cell_selectable=False,
            row_selectable='single',
        )

        # --- model selector: edit modal ---
        self._edit_modal = dbc.Modal(id=prefix + 'edit_modal', children=[
            dbc.ModalHeader(dbc.ModalTitle('Edit Tile Location Model')),
            dbc.ModalBody([
                html.P(id=prefix + 'edit_modal_textbox', className='text-danger'),
                html.P('Toggle Active to apply or remove this model, then press Confirm.',
                       className='mb-3'),
                dbc.Form([
                    dbc.Row([dbc.Label('Model File:', width=3),
                             dbc.Label(id=prefix + 'file_label', width=9)]),
                    dbc.Row([dbc.Label('File Exists:', width=3),
                             dbc.Label(id=prefix + 'file_exists_label', width=9)]),
                    dbc.Row([
                        dbc.Label('Active', html_for=prefix + 'is_active_input', width=3),
                        dbc.Col(dbc.Switch(id=prefix + 'is_active_input', value=True,
                                           label='Model is used for tile corner detection'), width=9),
                    ], className='mb-3'),
                ]),
                html.Div([
                    dbc.Button('Confirm', id={'type': prefix + 'edit_action', 'index': 'confirm'},
                               n_clicks=0, className='me-3'),
                    dbc.Button('Cancel', id={'type': prefix + 'edit_action', 'index': 'cancel'},
                               n_clicks=0, color='secondary'),
                ], className='text-center mt-3'),
            ]),
        ], size='lg', is_open=False)

        # --- parameter editor ---
        cur_ws = CONFIG.get(_PARAM_KEYS['working_scale'], _PARAM_DEFAULTS['working_scale'])
        cur_ts = CONFIG.get(_PARAM_KEYS['template_size'], _PARAM_DEFAULTS['template_size'])
        cur_cs = CONFIG.get(_PARAM_KEYS['corner_size'],   _PARAM_DEFAULTS['corner_size'])

        # Confirmation modal for parameter saves
        self._params_confirm_modal = dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle('Confirm Parameter Update')),
            dbc.ModalBody([
                html.P('The following loctile parameters will be updated and applied immediately:',
                       className='mb-2'),
                html.Ul([
                    html.Li(id=prefix + 'pcm_ws_li'),
                    html.Li(id=prefix + 'pcm_ts_li'),
                    html.Li(id=prefix + 'pcm_cs_li'),
                ]),
                html.P('This will take effect for the next detection run without restarting '
                       'the container.', className='text-info mt-2'),
            ]),
            dbc.ModalFooter([
                dbc.Button('Confirm', id={'type': prefix + 'params_action', 'index': 'confirm'},
                           n_clicks=0, className='me-2'),
                dbc.Button('Cancel', id={'type': prefix + 'params_action', 'index': 'cancel'},
                           n_clicks=0, color='secondary'),
            ]),
        ], id=prefix + 'params_confirm_modal', is_open=False)

        params_panel = dbc.Card([
            dbc.CardHeader(
                html.H6(dbc.Badge('TILE DETECTION PARAMETERS', color='white', text_color='secondary'),
                        className='mb-0')
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Working Scale'),
                        dbc.Input(id=prefix + 'ws_input', type='number',
                                  value=cur_ws, min=0.001, max=1.0, step=0.001),
                        dbc.FormFeedback('Must be a number in range (0, 1].', type='invalid'),
                        dbc.FormText('Downscale factor applied to images before corner matching.'),
                    ], width=4),
                    dbc.Col([
                        dbc.Label('Template Size (px)'),
                        dbc.Input(id=prefix + 'ts_input', type='number',
                                  value=cur_ts, min=1, step=1),
                        dbc.FormFeedback('Must be a positive integer.', type='invalid'),
                        dbc.FormText('Total template dimensions in pixels at working scale.'),
                    ], width=4),
                    dbc.Col([
                        dbc.Label('Corner Size (px)'),
                        dbc.Input(id=prefix + 'cs_input', type='number',
                                  value=cur_cs, min=1, step=1),
                        dbc.FormFeedback('Must be a positive integer less than Template Size.',
                                         type='invalid'),
                        dbc.FormText('Corner portion size; must be smaller than Template Size.'),
                    ], width=4),
                ], className='mb-3'),
                dbc.Button('Save Parameters', id=prefix + 'save_params_btn',
                           color='primary', size='sm'),
            ]),
        ], className='mt-3')

        # --- template type panel ---
        cur_tt = CONFIG.get(_PARAM_KEYS['template_type'], _PARAM_DEFAULTS['template_type'])

        template_type_panel = dbc.Card([
            dbc.CardHeader(
                html.H6(dbc.Badge('CORNER TEMPLATE TYPE', color='white', text_color='secondary'),
                        className='mb-0')
            ),
            dbc.CardBody([
                dbc.Alert(
                    [html.Strong('Block template: '), 'matches the large blue corner squares used in 2024 data. ',
                     html.Strong('Edge template: '), 'matches the thin black frame edges used from 2025 onwards. '
                     'Corner Size (px) only applies to the Edge template.'],
                    color='info', className='mb-3',
                ),
                dbc.Row([
                    dbc.Col(_template_preview_card(
                        'block', 'Block Template', 'Blue corners (2024)', 'primary'), width=6),
                    dbc.Col(_template_preview_card(
                        'edge', 'Edge Template', 'Black configuration (2025+)', 'success'), width=6),
                ], className='mb-3'),
                dbc.RadioItems(
                    id=prefix + 'template_type_radio',
                    options=[
                        {'label': 'Block Template — Blue corners (2024)', 'value': 'block'},
                        {'label': 'Edge Template — Black configuration (2025+)', 'value': 'edge'},
                    ],
                    value=cur_tt,
                    inline=True,
                    className='mb-3',
                ),
                dbc.Button('Save Template Type', id=prefix + 'save_tt_btn',
                           color='primary', size='sm'),
            ]),
        ], className='mt-3')

        # --- assemble panel ---
        self.the_panel = html.Div([
            html.H4(dbc.Badge('CURRENT TILE LOCATION MODELS', className='ms-1 me-2',
                              color='white', text_color='secondary')),
            dbc.Alert('Only one tile location model can be active at any one time. '
                      'Activating a model will replace the current selection.',
                      color='info', className='mt-2 mb-2'),
            dbc.Row(html.Div([
                dbc.Button('Edit', id={'type': prefix + 'table', 'index': 'update'},
                           n_clicks=0, color='secondary', className='mb-1 me-1', size='sm'),
                self._datatable,
            ], className='p-3 text-start', style={'background-color': 'rgb(225, 225, 225)'})),
            params_panel,
            template_type_panel,
            dcc.Store(id=prefix + 'row_edit_store'),
            dcc.Store(id=prefix + 'update_trigger'),
            dcc.Store(id=prefix + 'pending_params'),
            self._toast,
            self._edit_modal,
            self._confirm_modal,
            self._params_confirm_modal,
        ], className='text-center')

        # ---- callbacks ----

        # Table Edit button pressed → populate row_edit_store → triggers _edit_row_received
        self.app.callback(
            [Output(prefix + 'confirm_modal', 'is_open', allow_duplicate=True),
             Output(prefix + 'confirm_modal_title', 'children'),
             Output(prefix + 'confirm_modal_message', 'children'),
             Output(prefix + 'row_edit_store', 'data'),
             Output(prefix + 'datatable', 'selected_rows', allow_duplicate=True)],
            [State(prefix + 'datatable', 'data'),
             State(prefix + 'datatable', 'selected_rows'),
             Input({'type': prefix + 'table', 'index': ALL}, 'n_clicks')],
            prevent_initial_call=True,
        )(self._table_button_pressed())

        # row_edit_store updated → open edit modal and populate its fields
        self.app.callback(
            [Output(prefix + 'edit_modal', 'is_open', allow_duplicate=True),
             Output(prefix + 'file_label', 'children'),
             Output(prefix + 'file_exists_label', 'children'),
             Output(prefix + 'is_active_input', 'value'),
             Output(prefix + 'edit_modal_textbox', 'children')],
            [Input(prefix + 'row_edit_store', 'data')],
            prevent_initial_call=True,
        )(self._edit_row_received())

        # Confirm / Cancel inside edit modal
        self.app.callback(
            [Output(prefix + 'toast', 'is_open', allow_duplicate=True),
             Output(prefix + 'toast', 'children', allow_duplicate=True),
             Output(prefix + 'edit_modal', 'is_open', allow_duplicate=True),
             Output(prefix + 'update_trigger', 'data', allow_duplicate=True)],
            [State(prefix + 'is_active_input', 'value'),
             State(prefix + 'row_edit_store', 'data'),
             Input({'type': prefix + 'edit_action', 'index': ALL}, 'n_clicks')],
            prevent_initial_call=True,
        )(self._edit_row_confirmed())

        # Datatable refresh when update_trigger fires, or on initial page mount
        self.app.callback(
            [Output(prefix + 'datatable', 'data'),
             Output(prefix + 'datatable', 'selected_rows')],
            [Input(prefix + 'update_trigger', 'data')],
        )(self._refresh_datatable())

        # Save Parameters button → validate → open params confirmation modal
        self.app.callback(
            [Output(prefix + 'params_confirm_modal', 'is_open', allow_duplicate=True),
             Output(prefix + 'pcm_ws_li', 'children'),
             Output(prefix + 'pcm_ts_li', 'children'),
             Output(prefix + 'pcm_cs_li', 'children'),
             Output(prefix + 'pending_params', 'data'),
             Output(prefix + 'toast', 'is_open', allow_duplicate=True),
             Output(prefix + 'toast', 'children', allow_duplicate=True),
             Output(prefix + 'ws_input', 'invalid'),
             Output(prefix + 'ts_input', 'invalid'),
             Output(prefix + 'cs_input', 'invalid')],
            [Input(prefix + 'save_params_btn', 'n_clicks'),
             State(prefix + 'ws_input', 'value'),
             State(prefix + 'ts_input', 'value'),
             State(prefix + 'cs_input', 'value')],
            prevent_initial_call=True,
        )(self._validate_params())

        # Confirm / Cancel inside params confirmation modal → save or discard
        self.app.callback(
            [Output(prefix + 'params_confirm_modal', 'is_open', allow_duplicate=True),
             Output(prefix + 'toast', 'is_open', allow_duplicate=True),
             Output(prefix + 'toast', 'children', allow_duplicate=True)],
            [State(prefix + 'pending_params', 'data'),
             Input({'type': prefix + 'params_action', 'index': ALL}, 'n_clicks')],
            prevent_initial_call=True,
        )(self._params_confirmed())

        # Save Template Type button → write to config immediately
        self.app.callback(
            [Output(prefix + 'toast', 'is_open', allow_duplicate=True),
             Output(prefix + 'toast', 'children', allow_duplicate=True)],
            [Input(prefix + 'save_tt_btn', 'n_clicks'),
             State(prefix + 'template_type_radio', 'value')],
            prevent_initial_call=True,
        )(self._save_template_type())

    def get_panel(self):
        return self.the_panel

    # ---- callback implementations ----

    def _table_button_pressed(self):
        def table_button_pressed(model, selected_rows, *args):
            if not selected_rows:
                raise PreventUpdate
            row = model[selected_rows[0]]
            button_id = ctx.triggered_id if ctx.triggered_id is not None else {}
            if button_id.get('index', '').endswith('update'):
                return (False, None, None, row, [])
            return (False, None, None, None, [])
        return table_button_pressed

    def _edit_row_received(self):
        def edit_row_received(row):
            if row is None:
                raise PreventUpdate
            fname = row['Filename']
            filepath = os.path.join(_MODELS_DIR, fname)
            exists = os.path.isfile(filepath)
            is_active = row['Active'] == 'Yes'
            warning = '' if exists else 'Warning: model file not found on disk.'
            exists_str = 'Yes' if exists else 'No — file missing'
            return (True, filepath, exists_str, is_active, warning)
        return edit_row_received

    def _edit_row_confirmed(self):
        def edit_row_confirmed(is_active, row, *args):
            if row is None:
                raise PreventUpdate
            button_id = ctx.triggered_id if ctx.triggered_id is not None else {}
            if not button_id.get('index', '').endswith('confirm'):
                return (False, ' ', False, dash.no_update)
            fname = row['Filename']
            filepath = os.path.join(_MODELS_DIR, fname)
            if is_active and not os.path.isfile(filepath):
                return (True, f'Cannot activate: file does not exist — {fname}', True, dash.no_update)
            if is_active:
                ok = _save_active_model(fname)
                msg = f'Tile location model set to: {fname}' if ok \
                    else 'Failed to save config — key not found in config file'
                return (True, msg, False, True if ok else dash.no_update)
            else:
                ok = _save_active_model('')
                msg = 'Tile location model deactivated' if ok else 'Failed to save config'
                return (True, msg, False, True if ok else dash.no_update)
        return edit_row_confirmed

    def _refresh_datatable(self):
        def refresh_datatable(trigger):
            return (_list_available_models(), [])
        return refresh_datatable

    def _validate_params(self):
        def validate_params(n_clicks, ws, ts, cs):
            ws_invalid = ts_invalid = cs_invalid = False
            errors = []

            if ws is None or not isinstance(ws, (int, float)):
                ws_invalid = True
                errors.append('Working Scale must be a number.')
            elif not (0 < float(ws) <= 1.0):
                ws_invalid = True
                errors.append('Working Scale must be in range (0, 1].')

            if ts is None or not isinstance(ts, (int, float)):
                ts_invalid = True
                errors.append('Template Size must be a number.')
            elif ts < 1 or int(ts) != ts:
                ts_invalid = True
                errors.append('Template Size must be a positive integer.')

            if cs is None or not isinstance(cs, (int, float)):
                cs_invalid = True
                errors.append('Corner Size must be a number.')
            elif cs < 1 or int(cs) != cs:
                cs_invalid = True
                errors.append('Corner Size must be a positive integer.')
            elif not ts_invalid and cs >= ts:
                cs_invalid = True
                errors.append('Corner Size must be strictly less than Template Size.')

            if errors:
                return (False, dash.no_update, dash.no_update, dash.no_update,
                        None, True, ' | '.join(errors),
                        ws_invalid, ts_invalid, cs_invalid)

            pending = {
                _PARAM_KEYS['working_scale']: float(ws),
                _PARAM_KEYS['template_size']: int(ts),
                _PARAM_KEYS['corner_size']:   int(cs),
            }
            return (True,
                    f'Working Scale: {float(ws)}',
                    f'Template Size: {int(ts)} px',
                    f'Corner Size: {int(cs)} px',
                    pending,
                    False, dash.no_update,
                    False, False, False)
        return validate_params

    def _params_confirmed(self):
        def params_confirmed(pending, *args):
            button_id = ctx.triggered_id if ctx.triggered_id is not None else {}
            if not button_id.get('index', '').endswith('confirm'):
                return (False, dash.no_update, dash.no_update)
            if not pending:
                return (False, True, 'No pending parameters to save.')
            ok = _write_config_values(pending)
            if ok:
                ws = pending.get(_PARAM_KEYS['working_scale'], '?')
                ts = pending.get(_PARAM_KEYS['template_size'], '?')
                cs = pending.get(_PARAM_KEYS['corner_size'], '?')
                logger.info(f'LocTileModelSelectBlock: saved loctile params '
                            f'working_scale={ws} template_size={ts} corner_size={cs}')
                msg = f'Parameters saved: working_scale={ws}, template_size={ts}, corner_size={cs}'
            else:
                msg = 'Failed to write parameters to config file.'
            return (False, True, msg)
        return params_confirmed

    def _save_template_type(self):
        def save_template_type(n_clicks, tt_value):
            if not tt_value:
                return (True, 'No template type selected.')
            ok = _write_config_values({_PARAM_KEYS['template_type']: tt_value})
            if ok:
                label = 'Block (Blue corners, 2024)' if tt_value == 'block' else 'Edge (Black configuration, 2025+)'
                logger.info(f'LocTileModelSelectBlock: template type saved: {tt_value}')
                return (True, f'Template type saved: {label}')
            return (True, 'Failed to write template type to config file.')
        return save_template_type
