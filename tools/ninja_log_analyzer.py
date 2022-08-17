import logging
import argparse
from collections import namedtuple
from math import floor

logging.basicConfig(format="%(levelname)s: %(message)s")

Entry = namedtuple("Entry", "name hash start end duration")

def read_entries(path):
    """
    Read entries from a ninja log file.
    Only returns the entries from the last run.
    """
    try:
        f = open(path, "r")
    except:
        logging.error(f"Cannot open '{path}'!")
        return None

    # Check header.
    header = f.readline()
    if header != "# ninja log v5\n":
        logging.error(f"{path}:1 has invalid header!")
        return None

    # Read entries.
    entries = []
    last_end = None
    last_hash = None
    line_index = 1
    for line in f.readlines():
        line_index += 1
        tokens = line.strip().split("\t")
        if len(tokens) != 5:
            logging.warn(f"{path}:{line_index} contains invalid entry (found {len(tokens)} tokens instead of 5.")
        name = tokens[3]
        hash = tokens[4]
        start = int(tokens[0]) / 1000.0
        end = int(tokens[1]) / 1000.0
        duration = end - start
        # Reset entries if a new run is detected.
        if last_end and end < last_end:
            entries = []
        # Skip duplicate entries based on hash.
        if not last_hash or hash != last_hash:
            entries.append(Entry(name, hash, start, end, duration))
        last_end = end
        last_hash = hash
    return entries

def format_duration(duration):
    """
    Format a duration (seconds) into a string of the format minutes:seconds.
    """
    minutes = floor(duration) // 60
    seconds = duration - minutes * 60
    seconds = floor(seconds * 100) / 100
    return f"{minutes:02d}:{seconds:05.2f}"

def run(args):
    """
    Runs the logfile analyzer.
    Prints total build time followed by the longest entries.
    """
    print("Ninja log analyzer:")

    entries = read_entries(args.logfile)
    if entries == None:
        return
    if entries == []:
        print("Empty log file.")
        return
    entries = sorted(entries, key=lambda e: e.duration)

    global_start = min(entries, key=lambda e: e.start).start
    global_end = max(entries, key=lambda e: e.end).end
    global_duration = global_end - global_start

    print(f"{format_duration(global_duration)} Total build time")

    for e in reversed(entries[-args.count:]):
        print(f"{format_duration(e.duration)} {e.name}")
    pass


parser = argparse.ArgumentParser()
parser.add_argument("logfile", type=str, help="file path to the ninja logfile")
parser.add_argument("-c", "--count", type=int, default=10, help="number of log entries to show")
args = parser.parse_args()
run(args)
