'''
Script for viewing image test results.
'''

import sys
import time
import datetime
import threading
import re
import json
import argparse
import webbrowser
from pathlib import Path

import libs.bottle as bottle
from libs.bottle import route, view, request, run, template, static_file

from core import Environment, config, helpers

# Directory containing viewer files.
VIEWER_DIR = Path(__file__).parent / 'viewer'

# Directory containing static files.
STATIC_DIR = VIEWER_DIR / 'static'

# Create set of all static files.
STATIC_FILES = set(map(lambda f: f.relative_to(STATIC_DIR).as_posix(), STATIC_DIR.glob('*')))

# Directory containing templates.
TEMPLATE_DIR = VIEWER_DIR / 'views'

# Setup templates.
bottle.TEMPLATE_PATH = [TEMPLATE_DIR]

# Result tag titles.
TAG_TITLES = {
    'build_config': 'Build Config',
    'vcs_root': 'VCS Root',
    'hostname': 'Hostname',
    'build_id': 'Build ID',
    'branch': 'Branch'
}

# File extension -> MIME type mapping.
MIME_TYPES = {
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.wasm': 'application/wasm',
}

# Global database instance.
database = None

class Database:
    '''
    Helper for accessing image test results.
    '''
    def __init__(self, env):
        result_dir_template = env.image_tests_result_dir
        ref_dir_template = env.image_tests_ref_dir

        # Substitute project_dir as it is a static part of the path.
        result_dir_template = result_dir_template.replace('${project_dir}', str(env.project_dir))
        ref_dir_template = ref_dir_template.replace('${project_dir}', str(env.project_dir))

        # Extract result directory and run pattern.
        index = result_dir_template.index('$')
        result_dir = result_dir_template[0:index]
        run_pattern = result_dir_template[index:].replace('\\', '/')

        # Extract reference directory.
        index = ref_dir_template.index('$')
        ref_dir = ref_dir_template[0:index]

        # Extract the tags that make up a run directory.
        run_tags = run_pattern.split('/')
        run_tags = list(map(lambda f: re.fullmatch(r'\$\{([a-z_]+)\}', f)[1], run_tags))

        # Create glob pattern for collecting runs.
        run_glob = '/'.join(['*'] * len(run_tags)) + '/report.json'

        self.result_dir = Path(result_dir)
        self.ref_dir = Path(ref_dir)
        self.run_tags = run_tags
        self.run_glob = run_glob

    def run_report_file(self, run_dir):
        '''
        Return full path to run report file.
        '''
        return self.result_dir / run_dir / 'report.json'

    def test_report_file(self, run_dir, test_dir):
        '''
        Return full path to test report file.
        '''
        return self.result_dir / run_dir / test_dir / 'report.json'

    def load_runs(self):
        '''
        Load list of all runs (not including reports of individual tests).
        '''
        runs = []

        for run_report_file in self.result_dir.glob(self.run_glob):
            run = self.load_run(run_report_file, load_tests=False)
            if run:
                runs.append(run)

        # Sort by date.
        runs.sort(key=lambda r: r['date'], reverse=True)

        return runs

    def load_run(self, run_report_file, load_tests=True):
        '''
        Load a single run (including reports of individual tests by default).
        '''
        run_dir = run_report_file.parent.relative_to(self.result_dir).as_posix()
        run = load_json(run_report_file)
        if not run:
            return None

        run_values = str(run_dir).split('/')
        run_tags = { k:v for [k, v] in zip(self.run_tags, run_values)}
        run['run_dir'] = str(run_dir)
        run['run_tags'] = run_tags

        if load_tests:
            tests = []

            for test_report_file in (self.result_dir / run_dir).glob('*/**/report.json'):
                test = self.load_test(test_report_file, load_log=False)
                if test:
                    tests.append(test)

            run['tests'] = tests

        return run

    def load_test(self, test_report_file, load_log=True):
        '''
        Load a single test (including test log file by default).
        '''
        test = load_json(test_report_file)
        if not test:
            return None

        if load_log:
            log_file = test_report_file.parent / 'log.txt'
            if log_file.exists():
                test['log'] = log_file.read_text('utf-8')

        return test


def load_json(path):
    '''
    Load a JSON file or return None if not successful.
    '''
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def run_stats(run):
    '''
    Compute stats bar data for a run.
    '''
    total_count = len(run['tests'])
    if total_count == 0:
        return []
    stats = []
    for result, color in zip(['PASSED', 'SKIPPED', 'FAILED'], ['#32b643', '#ffb700', '#e85600']):
        count = len(list(filter(lambda test: test['result'] == result, run['tests'])))
        stats.append({
            'title': result,
            'percentage': round(100 * count / total_count, 1),
            'color': color
        })
    return stats

def test_stats(test):
    '''
    Compute stats bar data for a test.
    '''
    total_count = len(test['images'])
    if total_count == 0:
        return []
    stats = []
    for result, color in zip(['PASSED', 'FAILED'], ['#32b643', '#e85600']):
        count = len(list(filter(lambda i: i['success'] == (result == 'PASSED'), test['images'])))
        stats.append({
            'title': result,
            'percentage': round(100 * count / total_count, 1),
            'color': color
        })
    return stats

def create_jeri_data(result_image, ref_image, error_image, extra_metrics=['L1', 'L2', 'MAPE', 'MRSE', 'SMAPE', 'SSIM']):
    '''
    Create a jeri config object for comparing two images.
    '''
    jeri_data = {
        'title': 'root',
        'children': [
            {
                'title': 'Result',
                'image': str(result_image)
            },
            {
                'title': 'Reference',
                'image': str(ref_image)
            },
            {
                'title': 'Error',
                'image': str(error_image),
                'tonemapGroup': 'error'
            }
        ]
    }

    for metric in extra_metrics:
        jeri_data['children'].append(
            {
                'title': metric,
                'lossMap': {
                    'function': metric,
                    'imageA': str(ref_image),
                    'imageB': str(result_image)
                },
                'tonemapGroup': 'metric'
            }
        )

    return jeri_data


def format_date(date):
    '''
    Convert a date in iso format to a human readable string.
    '''
    return datetime.datetime.fromisoformat(date).strftime('%d. %B %Y %I:%M%p')

def format_duration(duration):
    '''
    Convert a duration in seconds to a human readable string.
    '''
    return str(datetime.timedelta(seconds=round(duration)))



# routes
# / - list of runs
# /run_id - display selected run
# /run_id/test_id - display selected test
# /run_id/test_id?action=compare,image=xxx - compare single image

# /<path to file> - static file
# /result/<path to image> - result images
# /ref/<path to image> - reference images

def parse_path(path):
    '''
    Parses a path and returns a tuple containing the run and test directories.
    Paths are expected to have the form <run_dir>/<test_dir>.
    '''
    tags = path.split('/')

    run_dir = None
    test_dir = None

    run_tag_count = len(database.run_tags)
    if len(tags) >= run_tag_count:
        run_dir = '/'.join(tags[0:run_tag_count])
        tags = tags[run_tag_count:]
    else:
        tags = []

    if len(tags) > 0:
        test_dir = '/'.join(tags)

    return run_dir, test_dir


@route('/result/<path:path>')
def result_image(path):
    return static_file(path, database.result_dir)

@route('/ref/<path:path>')
def result_image(path):
    return static_file(path, database.ref_dir)

@route('/')
@view('index')
def index_page():
    runs = database.load_runs()
    nav = [
        { 'title': 'Home', 'link': '/'}
    ]
    return dict(
        nav=nav,
        tag_titles=TAG_TITLES,
        hostname=helpers.get_hostname(),
        result_dir=database.result_dir,
        runs=runs,
        run_tags=database.run_tags,
        format_date=format_date,
        format_duration=format_duration
    )

@route('/<path:path>')
def catch_all(path):
    # Return static files.
    if path in STATIC_FILES:
        mimetype = MIME_TYPES.get(Path(path).suffix, 'auto')
        return static_file(path, STATIC_DIR, mimetype=mimetype)

    # Parse path.
    run_dir, test_dir = parse_path(path)

    if run_dir and not test_dir:
        # Show run page.
        run = database.load_run(database.run_report_file(run_dir))
        if not run:
            return template('error', message=f'Run "{run_dir}" does not exist.')

        nav = [
            { 'title': 'Home', 'link': '/'},
            { 'title': 'Run: ' + run_dir, 'link': '/' + run_dir }
        ]
        stats = run_stats(run)
        return template(
            'run',
            nav=nav,
            tag_titles=TAG_TITLES,
            stats=stats,
            run_dir=run_dir,
            run=run,
            run_tags=database.run_tags,
            format_date=format_date,
            format_duration=format_duration
        )

    elif run_dir and test_dir:
        # Show test page.
        test = database.load_test(database.test_report_file(run_dir, test_dir))
        if not test:
            return template('error', message=f'Test "{test_dir}" does not exist for run "{run_dir}".')

        action = request.query.get('action', None)
        if action == 'compare':
            # Compare images.
            image = Path(request.query['image']).as_posix()
            result_image = Path('/result') / run_dir / test_dir / image
            error_image = Path(str(result_image) + config.ERROR_IMAGE_SUFFIX)
            ref_dir = Path(test['ref_dir']).relative_to(database.ref_dir)
            ref_image = Path('/ref') / ref_dir / image
            jeri_data = create_jeri_data(result_image, ref_image, error_image)
            return template(
                'compare',
                image=str(image),
                jeri_data=json.dumps(jeri_data)
            )
        else:
            nav = [
                { 'title': 'Home', 'link': '/'},
                { 'title': 'Run: ' + run_dir, 'link': '/' + run_dir },
                { 'title': 'Test: ' + test_dir, 'link': '/' + run_dir + '/' + test_dir }
            ]
            stats = test_stats(test)
            ref_dir = str(Path(test['ref_dir']).relative_to(database.ref_dir).as_posix())
            return template(
                'test',
                nav=nav,
                stats=stats,
                run_dir=run_dir,
                test_dir=test_dir,
                ref_dir=ref_dir,
                test=test,
                format_duration=format_duration
            )
    else:
        return template('error', message='Invalid URL.')

    return str([run_dir, test_dir])


def main():
    parser = argparse.ArgumentParser(description="Utility for viewing results of image tests.")
    parser.add_argument('-e', '--environment', type=str, action='store', help='Environment', default=config.DEFAULT_ENVIRONMENT)
    parser.add_argument('--host', type=str, action='store', help='Server hostname', default='localhost')
    parser.add_argument('--port', type=int, action='store', help='Server port', default=8080)
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser window')

    args = parser.parse_args()

    # Load environment.
    try:
        env = Environment(args.environment, config.DEFAULT_BUILD_CONFIG)
    except Exception as e:
        print(e)
        sys.exit(1)

    # Create database.
    global database
    database = Database(env)

    url = f'http://{args.host}:{args.port}'
    print(f'Running server on {url}')

        # Open browser window.
    if not args.no_browser:
        webbrowser.open(url)

    # Trampoline function for running server.
    def run_server():
        run(host=args.host, port=args.port, debug=args.debug, reloader=args.debug, quiet=not args.debug)

    # Start server thread.
    server_thread = threading.Thread(target=run_server, name='server', daemon=True)
    server_thread.start()

    # Hack to handle KeyboardInterrupt and SystemExit.
    try:
        while server_thread.is_alive():
            server_thread.join(1)
    except (KeyboardInterrupt, SystemExit):
        print("Terminating...")

if __name__ == '__main__':
    main()
