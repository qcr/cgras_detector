# Copyright 2024 - Andrew Kwok Fai LUI,
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

from dash import html
import dash_bootstrap_components as dbc

# Helpers ----------------------------------------------------------------

def _figure(src, caption):
    """Responsive image with an italic caption below."""
    return html.Figure([
        html.Img(src=src, style={'width': '100%', 'height': 'auto', 'borderRadius': '6px'}),
        html.Figcaption(caption, className='text-muted small fst-italic mt-1 text-center'),
    ], style={'margin': 0})


def _stat(value, label, colour):
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.Div(value, className='fw-bold fs-3 mb-1', style={'color': colour}),
            html.P(label, className='text-muted small mb-0'),
        ], className='text-center'), className='h-100 border-0 shadow-sm'),
        md=3, sm=6, className='mb-3'
    )


class AboutPage():
    def __init__(self, app):
        self.app = app
        self._build_layout()

    def layout(self):
        return self._layout

    def _build_layout(self):

        # ── Hero ─────────────────────────────────────────────────────────────
        hero = dbc.Card(
            dbc.CardBody([
                html.H2('Coral Growout Robotic Assessment System', className='mb-1 fw-bold'),
                html.H4('CGRAS', className='mb-3', style={'color': '#e67e22'}),
                html.P(
                    'An integrated robotic imaging and AI platform for automated, high-throughput '
                    'monitoring of coral recruits in aquaculture facilities — accelerating reef '
                    'restoration at the scale the Great Barrier Reef needs.',
                    className='lead mb-0'
                ),
            ]),
            className='mb-4 border-0 shadow-sm',
            style={'backgroundColor': '#fff8f0'}
        )

        # ── The Challenge ────────────────────────────────────────────────────
        challenge = dbc.Card([
            dbc.CardHeader(html.H5('The Challenge', className='mb-0 fw-bold')),
            dbc.CardBody([
                html.P([
                    'Climate-driven mass bleaching is devastating the Great Barrier Reef, creating '
                    'urgent demand for scalable restoration. ',
                    html.Em('Coral seeding'), ' — growing recruits from spawn in aquaculture '
                    'facilities and transplanting them onto damaged reefs — is a proven, scalable '
                    'strategy endorsed by the ',
                    html.Strong('Reef Restoration and Adaptation Program (RRAP)'),
                    ', the world\'s largest coordinated reef restoration initiative.'
                ]),
                html.P([
                    'Effective growout requires consistent, high-frequency monitoring of thousands '
                    'of microscopic coral recruits growing on each concrete tile. A single tile '
                    'assessment by a trained expert takes up to ',
                    html.Strong('1 hour of microscope work'),
                    '. Facilities housing more than 10,000 devices per week cannot be manually '
                    'monitored at restoration scale — automation is essential.'
                ], className='mb-0'),
            ]),
        ], className='mb-4 shadow-sm')

        # ── What CGRAS Is — with process diagram ─────────────────────────────
        what = dbc.Card([
            dbc.CardHeader(html.H5('What is CGRAS?', className='mb-0 fw-bold')),
            dbc.CardBody([
                html.P(
                    'CGRAS is a scalable, reliable, and cost-effective solution for coral health '
                    'monitoring during the growout process. It consists of two tightly integrated '
                    'subsystems: the Image Acquisition System (IAS) and the Coral Counting & '
                    'Visualisation System (CCVS).',
                    className='mb-4'
                ),
                # Process-overview diagram — full width
                _figure(
                    '/assets/images/about_process_overview.png',
                    'Fig. 1 — CGRAS pipeline: coral larvae settle onto tiles in monitoring tanks; '
                    'the robot arm captures a 6×4 image grid per tile; images are stitched, '
                    'fed to a YOLO detector, and aggregated into a full-tile heatmap. '
                    'Live survival data then informs tank condition adjustments throughout growout.'
                ),
            ]),
        ], className='mb-4 shadow-sm')

        # ── Hardware — closeup + wide installation ────────────────────────────
        hardware = dbc.Card([
            dbc.CardHeader(html.H5('The Hardware', className='mb-0 fw-bold')),
            dbc.CardBody([
                dbc.Row([
                    # IAS close-up (portrait)
                    dbc.Col(
                        _figure(
                            '/assets/images/about_hw_closeup.png',
                            'Fig. 2 — IAS end-effector: robotic arm, IP-rated enclosure, '
                            'coral camera, depth camera, and LED ring lights.'
                        ),
                        md=4, className='mb-3 mb-md-0'
                    ),
                    # Full installation (landscape)
                    dbc.Col([
                        _figure(
                            '/assets/images/about_hw_overview.png',
                            'Fig. 3 — Full CGRAS installation at the AIMS National Sea Simulator '
                            '(SeaSim). The UR20 robotic arm traverses between two coral growth '
                            'tanks, imaging up to 50 tiles per session.'
                        ),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader(
                                        html.H6('IAS — Image Acquisition System',
                                                className='mb-0 fw-semibold text-white'),
                                        style={'backgroundColor': '#2e7d6b'}
                                    ),
                                    dbc.CardBody([
                                        html.P([
                                            'A ', html.Strong('Universal Robots UR20'), ' arm '
                                            'carries a waterproof end-effector housing a ',
                                            html.Strong('Sony ILX-LR1 camera'),
                                            ' (9,428 × 6,309 px, 50 mm macro lens) and LED ring '
                                            'illumination (≥3,000 lm). The arm follows a '
                                            'predefined open-loop path to capture a 6 × 4 grid '
                                            'of overlapping images per tile, stitched into a '
                                            'single ~28,000 × 28,000 px tile image.'
                                        ], className='mb-0 small'),
                                    ]),
                                ], className='border-0 shadow-sm'),
                            ], md=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader(
                                        html.H6('CCVS — Coral Counting & Visualisation System',
                                                className='mb-0 fw-semibold text-white'),
                                        style={'backgroundColor': '#1a5276'}
                                    ),
                                    dbc.CardBody([
                                        html.Ul([
                                            html.Li('Image stitching of 24 captures per tile'),
                                            html.Li('Tile frame localisation (pixel classifier + template matching)'),
                                            html.Li('YOLOv8n detection across 640 × 640 patches'),
                                            html.Li('Alive / dead classification per species'),
                                            html.Li('Spatial heatmaps and survival trend charts'),
                                            html.Li('This web interface — visualisation & export'),
                                        ], className='mb-0 small'),
                                    ]),
                                ], className='border-0 shadow-sm'),
                            ], md=6),
                        ], className='g-2 mt-3'),
                    ], md=8),
                ]),
            ]),
        ], className='mb-4 shadow-sm')

        # ── Detection results ────────────────────────────────────────────────
        detections = dbc.Card([
            dbc.CardHeader(html.H5('Coral Detection in Action', className='mb-0 fw-bold')),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(
                        _figure(
                            '/assets/images/about_detections.png',
                            'Fig. 4 — Close-up of automated coral recruit detections on a tile '
                            'sub-region. Each bounding box shows the detected class '
                            '(POLYP_MULTI = multi-polyp colony) and confidence score. '
                            'Red grid lines indicate the 14 × 14 mm tab boundaries.'
                        ),
                        md=7
                    ),
                    dbc.Col([
                        html.H6('How detection works', className='fw-semibold mt-2 mt-md-0'),
                        html.P([
                            'Because the full stitched tile image is far too large for direct '
                            'YOLO inference (~28,000 × 28,000 px), the CCVS adopts a ',
                            html.Strong('sliced inference strategy'),
                            ': the image is divided into overlapping 640 × 640 px patches, '
                            'each processed independently by a ',
                            html.Strong('YOLOv8n'), ' object detection model (3.2 M parameters). '
                            'Detections are then de-duplicated across overlapping regions '
                            'and merged back into the full tile coordinate space.'
                        ]),
                        html.P([
                            'The model distinguishes two coral health classes — ',
                            html.Strong('alive'), ' and ', html.Strong('dead'),
                            ' — across multiple species, and is trained on manually annotated '
                            'tile images using a bootstrapped semi-supervised workflow that '
                            'reduces labelling time by ~5× per image.'
                        ]),
                        dbc.Row([
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.Div('89.1%', className='fw-bold fs-4', style={'color': '#2874a6'}),
                                html.P('Precision', className='text-muted small mb-0'),
                            ], className='text-center'), className='border-0 shadow-sm'), width=4),
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.Div('85.9%', className='fw-bold fs-4', style={'color': '#2874a6'}),
                                html.P('Recall', className='text-muted small mb-0'),
                            ], className='text-center'), className='border-0 shadow-sm'), width=4),
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.Div('87.5%', className='fw-bold fs-4', style={'color': '#2874a6'}),
                                html.P('F1 Score', className='text-muted small mb-0'),
                            ], className='text-center'), className='border-0 shadow-sm'), width=4),
                        ], className='g-2'),
                    ], md=5, className='d-flex flex-column justify-content-center'),
                ], className='align-items-center'),
            ]),
        ], className='mb-4 shadow-sm')

        # ── Deployment ───────────────────────────────────────────────────────
        deployment = dbc.Card([
            dbc.CardHeader(html.H5('Deployment', className='mb-0 fw-bold')),
            dbc.CardBody([
                html.P([
                    'CGRAS was deployed at the ',
                    html.Strong(
                        'Australian Institute of Marine Science (AIMS) '
                        'National Sea Simulator (SeaSim)'
                    ),
                    ', an indoor aquaculture facility where corals are grown under controlled '
                    'conditions. Coral mass spawning on the Great Barrier Reef occurs only once '
                    'per year; CGRAS was operational during the ',
                    html.Strong('2024 and 2025 spawning events'),
                    ', monitoring the primary restoration species:'
                ]),
                html.Ul([
                    html.Li([html.Em('Acropora kenti'), ' (Aken) — fast-growing, widely used in restoration']),
                    html.Li([html.Em('Acropora millepora'), ' (Amil) — key restoration species, shared early morphology with Aken']),
                    html.Li([html.Em('Platygyra daedalea'), ' (Pdae) — distinct morphology, included for model generalisation testing']),
                ]),
                html.P([
                    'During the 2025 growout period, CGRAS completed ',
                    html.Strong('183 tile imaging sessions'),
                    ', with 99.5% completed without error and an 87.9% CCVS processing yield.'
                ], className='mb-0'),
            ]),
        ], className='mb-4 shadow-sm')

        # ── Impact metrics ───────────────────────────────────────────────────
        impact = dbc.Card([
            dbc.CardHeader(html.H5('Measured Impact', className='mb-0 fw-bold')),
            dbc.CardBody([
                dbc.Row([
                    _stat('12.3×',   'overall speed-up vs. equivalent manual effort',         '#e67e22'),
                    _stat('4.8 min', 'imaging time per tile  (down from 30 min)',             '#2e7d6b'),
                    _stat('7.3 min', 'counting time per tile  (down from 1 hour)',            '#1a5276'),
                    _stat('95%',     'counting accuracy vs. expert manual assessment',        '#7d3c98'),
                ], className='mb-2'),
                dbc.Row([
                    _stat('183',    'tile imaging sessions in 2025 growout period',           '#2e7d6b'),
                    _stat('99.5%',  'of imaging sessions completed without error',            '#2e7d6b'),
                    _stat('87.9%',  'CCVS processing yield across all sessions',              '#1a5276'),
                    dbc.Col(
                        dbc.Card(dbc.CardBody([
                            html.Div('RRAP', className='fw-bold fs-3 mb-1',
                                     style={'color': '#b7950b'}),
                            html.P(
                                "Contributing to the world's largest coordinated reef "
                                "restoration program",
                                className='text-muted small mb-0'
                            ),
                        ], className='text-center'), className='h-100 border-0 shadow-sm'),
                        md=3, sm=6, className='mb-3'
                    ),
                ]),
            ]),
        ], className='mb-4 shadow-sm')

        # ── Team / Acknowledgements ──────────────────────────────────────────
        team = dbc.Card([
            dbc.CardHeader(html.H5('Team & Acknowledgements', className='mb-0 fw-bold')),
            dbc.CardBody([
                html.P([
                    'CGRAS is developed by the ',
                    html.Strong(
                        'Robotics and Autonomous Systems Group, Research Engineering Facility (REF), '
                        'Queensland University of Technology (QUT)'
                    ),
                    ', in partnership with the ',
                    html.Strong('Australian Institute of Marine Science (AIMS)'),
                    ' and supported by the ',
                    html.Strong('Reef Restoration and Adaptation Program (RRAP)'),
                    ', funded by a partnership between the Australian Government\'s Reef Trust '
                    'and the Great Barrier Reef Foundation.'
                ]),
                html.P([
                    html.Strong('Authors: '),
                    'Dorian Tsai, Emilio Olivastri, Riki Lamont, Andrew Lui, Timothy Morris, '
                    'Joshua Esplin, Christopher A Brunner, Mikaela Nordborg, Reginald Warldeworth, '
                    'Garima Samvedi, Karen Jackel, Andrea Severati, Tobias Fischer, Scarlett Raine.'
                ]),
                html.P([
                    'The authors acknowledge the Traditional Owners of the land and sea Country '
                    'where this research was performed — the Manbarra, Wulgurukaba and Bindal '
                    'peoples — as the first scientists and custodians of their lands, and pay '
                    'respect to Elders past, present and emerging. The adult corals and coral '
                    'spawn used in this study were collected from traditional sea Country of the '
                    'Manbarra and Bindal peoples with their Free Prior and Informed Consent.'
                ], className='mb-0 fst-italic text-muted small'),
            ]),
        ], className='mb-5 shadow-sm')

        # ── Assemble ─────────────────────────────────────────────────────────
        self._layout = dbc.Container([
            dbc.Row(html.H3('About CGRAS', className='mt-3 mb-3')),
            hero,
            challenge,
            what,
            hardware,
            detections,
            deployment,
            impact,
            team,
        ], fluid=True)
