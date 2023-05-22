import re
import sys
import datetime
import time

from xml.dom.minidom import Document


__all__ = ('TestXMLBuilder', 'TestXMLContext')


# see issue #74, the encoding name needs to be one of
# http://www.iana.org/assignments/character-sets/character-sets.xhtml
UTF8 = 'UTF-8'

# Workaround for Python bug #5166
# http://bugs.python.org/issue5166

_char_tail = ''

if sys.maxunicode > 0x10000:
    _char_tail = (u'%s-%s') % (
        chr(0x10000),
        chr(min(sys.maxunicode, 0x10FFFF))
    )

_nontext_sub = re.compile(
    r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD%s]' % _char_tail,
    re.U
).sub


def replace_nontext(text, replacement=u'\uFFFD'):
    return _nontext_sub(replacement, text)


class TestXMLContext(object):
    """A XML report file have a distinct hierarchy. The outermost element is
    'testsuites', which contains one or more 'testsuite' elements. The role of
    these elements is to give the proper context to 'testcase' elements.

    These contexts have a few things in common: they all have some sort of
    counters (i.e. how many testcases are inside that context, how many failed,
    and so on), they all have a 'time' attribute indicating how long it took
    for their testcases to run, etc.

    The purpose of this class is to abstract the job of composing this
    hierarchy while keeping track of counters and how long it took for a
    context to be processed.
    """

    # Allowed keys for self.counters
    _allowed_counters = ('tests', 'errors', 'failures', 'skipped',)

    def __init__(self, xml_doc, parent_context=None):
        """Creates a new instance of a root or nested context (depending whether
        `parent_context` is provided or not).
        """
        self.xml_doc = xml_doc
        self.parent = parent_context
        self._start_time_m = 0
        self._stop_time_m = 0
        self._stop_time = 0
        self.counters = {}

    def element_tag(self):
        """Returns the name of the tag represented by this context.
        """
        return self.element.tagName

    def begin(self, tag, name):
        """Begins the creation of this context in the XML document by creating
        an empty tag <tag name='param'>.
        """
        self.element = self.xml_doc.createElement(tag)
        self.element.setAttribute('name', replace_nontext(name))
        self._start_time = time.monotonic()

    def end(self):
        """Closes this context (started with a call to `begin`) and creates an
        attribute for each counter and another for the elapsed time.
        """
        # time.monotonic is reliable for measuring differences, not affected by NTP
        self._stop_time_m = time.monotonic()
        # time.time is used for reference point
        self._stop_time = time.time()
        self.element.setAttribute('time', self.elapsed_time())
        self.element.setAttribute('timestamp', self.timestamp())
        self._set_result_counters()
        return self.element

    def _set_result_counters(self):
        """Sets an attribute in this context's tag for each counter considering
        what's valid for each tag name.
        """
        tag = self.element_tag()

        for counter_name in TestXMLContext._allowed_counters:
            valid_counter_for_element = False

            if counter_name == 'skipped':
                valid_counter_for_element = (
                    tag == 'testsuite'
                )
            else:
                valid_counter_for_element = (
                    tag in ('testsuites', 'testsuite')
                )

            if valid_counter_for_element:
                value = str(
                    self.counters.get(counter_name, 0)
                )
                self.element.setAttribute(counter_name, value)

    def increment_counter(self, counter_name):
        """Increments a counter named by `counter_name`, which can be any one
        defined in `_allowed_counters`.
        """
        if counter_name in TestXMLContext._allowed_counters:
            self.counters[counter_name] = \
                self.counters.get(counter_name, 0) + 1

    def elapsed_time(self):
        """Returns the time the context took to run between the calls to
        `begin()` and `end()`, in seconds.
        """
        return format(self._stop_time_m - self._start_time_m, '.3f')

    def timestamp(self):
        """Returns the time the context ended as ISO-8601-formatted timestamp.
        """
        return datetime.datetime.fromtimestamp(self._stop_time).replace(microsecond=0).isoformat()


class TestXMLBuilder(object):
    """This class encapsulates most rules needed to create a XML test report
    behind a simple interface.
    """

    def __init__(self):
        """Creates a new instance.
        """
        self._xml_doc = Document()
        self._current_context = None

    def current_context(self):
        """Returns the current context.
        """
        return self._current_context

    def begin_context(self, tag, name):
        """Begins a new context in the XML test report, which usually is defined
        by one on the tags 'testsuites', 'testsuite', or 'testcase'.
        """
        context = TestXMLContext(self._xml_doc, self._current_context)
        context.begin(tag, name)

        self._current_context = context

    def context_tag(self):
        """Returns the tag represented by the current context.
        """
        return self._current_context.element_tag()

    def _create_cdata_section(self, content):
        """Returns a new CDATA section containing the string defined in
        `content`.
        """
        filtered_content = replace_nontext(content)
        return self._xml_doc.createCDATASection(filtered_content)

    def append_cdata_section(self, tag, content):
        """Appends a tag in the format <tag>CDATA</tag> into the tag represented
        by the current context. Returns the created tag.
        """
        element = self._xml_doc.createElement(tag)

        pos = content.find(']]>')
        while pos >= 0:
            tmp = content[0:pos+2]
            element.appendChild(self._create_cdata_section(tmp))
            content = content[pos+2:]
            pos = content.find(']]>')

        element.appendChild(self._create_cdata_section(content))

        self._append_child(element)
        return element

    def append(self, tag, content, **kwargs):
        """Apends a tag in the format <tag attr='val' attr2='val2'>CDATA</tag>
        into the tag represented by the current context. Returns the created
        tag.
        """
        element = self._xml_doc.createElement(tag)

        for key, value in kwargs.items():
            filtered_value = replace_nontext(str(value))
            element.setAttribute(key, filtered_value)

        if content:
            element.appendChild(self._create_cdata_section(content))

        self._append_child(element)
        return element

    def _append_child(self, element):
        """Appends a tag object represented by `element` into the tag
        represented by the current context.
        """
        if self._current_context:
            self._current_context.element.appendChild(element)
        else:
            self._xml_doc.appendChild(element)

    def increment_counter(self, counter_name):
        """Increments a counter in the current context and their parents.
        """
        context = self._current_context

        while context:
            context.increment_counter(counter_name)
            context = context.parent

    def end_context(self):
        """Ends the current context and sets the current context as being the
        previous one (if it exists). Also, when a context ends, its tag is
        appended in the proper place inside the document.
        """
        if not self._current_context:
            return False

        element = self._current_context.end()

        self._current_context = self._current_context.parent
        self._append_child(element)

        return True

    def finish(self):
        """Ends all open contexts and returns a pretty printed version of the
        generated XML document.
        """
        while self.end_context():
            pass
        return self._xml_doc.toprettyxml(indent='\t', encoding=UTF8)
