import requests
import re
import html
import logging

class BaseExtractor():
    def __init__(self):

        #logger setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False  # Prevent logging from propagating to the root logger

    
    #----------------------------------------------------------
    #change logging level
    def set_logger_level(self, level):
        """Change the logging level for the logger and its handlers."""
        if isinstance(level, int) or level in logging._nameToLevel:
            self.logger.setLevel(level)
            for handler in self.logger.handlers:
                handler.setLevel(level)
        else:
            raise ValueError(f"Invalid logging level: {level}. Use one of {list(logging._nameToLevel.keys())}.")
        return
    

    #----------------------------------------------------------
    #fetches the raw html webpage
    def fetch_webpage(self, url):
        """Fetches raw HTML content from a given URL."""
        try: 
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36'}
            response = requests.get(url=url, headers=headers)

            if response.status_code == 200:
                return response.text
            else:
                self.logger.warning(f'failed with status code: {response.status_code}')
                return None
        except Exception as e:
            self.logger.error(f'error fetching webpage: {e}')
            return None

    #helper function for extracting from html
    def strip_tag(self, content, tag, extra_html=''):
        """
        Extracts content inside specified HTML tags from the given HTML string.
        
        Args:
            content (str): The HTML content to search.
            tag (str): The HTML tag name to extract.
            extra_html (str): Additional tag attributes to match (optional).
        
        Returns:
            list: A list of strings containing the matched content.
        """
        extra_html = re.escape(extra_html).replace(r'\=', '=').replace(r'\"', '"').replace(r"\'", "'") #sanitize extra html if necessary

        matches = re.findall(rf'<{tag}\s*{extra_html}[^>]*>(.*?)</{tag}>', content)
        return matches
    
    def remove_tag(self, content, tag):
        """
        Removes everything inside the tag, including the tags themselves.

        Args:
            content (str): The HTML content to clean.
            tag (str): The HTML tag name to remove.

        Returns:
            str: The cleaned content with tag removed.
        """
        cleaned = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', '', content, flags=re.DOTALL)
        return cleaned.strip()
    
    def select_regex(self, content, regex):
        """
        Selects specifically what is specified by the regex.

        Args:
            content (str): The HTML content to clean.
            regex (re str): Regular expression.

        Returns:
            str: The cleaned content with selected text, if nothing is found does nothing.
        """
        cleaned = re.search(regex, content)

        if cleaned:
            return cleaned.group(1)
        return content

    
    #----------------------------------------------------------
    #get the headers
    def extract_table_data(self, content, section='head'):
        """
        Extract data from a specific section of an HTML table (head or body).
        
        Args:
            content (str): HTML content to parse.
            section (str): If section is 'head', set tags to ('thead', 'tr', 'th'),
            otherwise set tags to ('tbody', 'tr', 'td')

        Returns:
            list: A nested list where each sublist represents a row of the table.
        """
        if section == 'head':
            section_tag, row_tag, cell_tag = ('thead', 'tr', 'th')
        else:
            section_tag, row_tag, cell_tag = ('tbody', 'tr', 'td')

        section_data = []
        sections = self.strip_tag(content, section_tag) #select all table headers

        for section in sections:
            row_data = []
            rows = self.strip_tag(section, row_tag) #select all rows inside each header
            
            for row in rows:
                cells = self.strip_tag(row, cell_tag) #select all entries in each row
                row_data.append(cells)

            section_data.append(row_data)

        return section_data
    
    #strips of as many html tags as possible selecting the most likely data
    def clean_string(self, content, strip_tags=['span', 'a'], remove_tags=['svg', 'img'], select_index=None):

        #removes any extra tags that would be outside the data
        for tag in remove_tags:
            content = self.remove_tag(content, tag)

        while True:
            stripped_any = False

            for tag in strip_tags:
                stripped = self.strip_tag(content, tag)

                if stripped:

                    if select_index is not None:
                        content = stripped[min(select_index, len(stripped)-1)]
                    else:
                        content = max(stripped, key=len) #most likely span to contain the data we are looking for
                    stripped_any = True
                    break #now redo it with the stripped part
            
            if not stripped_any: #exit loop when nothing left to strip
                break
        
        content = html.unescape(content) #unescape any html special characters
        return content
    

    def clean_table(self, table, strip_tags=['span', 'a'], remove_tags=['svg'], ignore_columns=[]):
        """
        Applies a function to each cell in a 2D list representing rows and cells.

        Args:
            table (list): A 2D list where each element is a row, which contain cells.
            strip_tags (list): List of elements to pass to clean_string.
            remove_tags (list): List of elements to pass to clean_string.
            ignore_columns (list): 0-indexed columns to ignore cleaning.

        Returns:
            list: A 2D list with the function applied to each cell.
        """
        for row_index, row in enumerate(table):
            for cell_index, cell in enumerate(row):
                #cleaning
                if cell_index not in ignore_columns:
                    table[row_index][cell_index] = self.clean_string(cell, strip_tags=strip_tags, remove_tags=remove_tags)
        return table

    def clean_tables(self, tables, strip_tags=['span', 'a'], remove_tags=['svg'], ignore_columns=[]):
        """
        Applies a function to each cell in a 3D list representing tables, rows, and cells.

        Args:
            tables (list): A 3D list where each element is a table, which contains rows, which contain cells.
            strip_tags (list): List of elements to pass to clean_string.
            remove_tags (list): List of elements to pass to clean_string.
            ignore_columns (list): 0-indexed columns to ignore cleaning.

        Returns:
            list: A 3D list with the function applied to each cell.
        """
        for table_index, table in enumerate(tables):
            tables[table_index] = self.clean_table(table, strip_tags=strip_tags, remove_tags=remove_tags, ignore_columns=ignore_columns)
        return tables

    #----------------------------------------------------------
    def extract_list_data():
        pass