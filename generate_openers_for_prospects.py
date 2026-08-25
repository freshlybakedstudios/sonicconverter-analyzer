#!/usr/bin/env python3
"""
Generate cold openers for regression_prospects who haven't been contacted yet.
"""

import os
import json
import time
import requests as _requests
from dotenv import load_dotenv
from supabase import create_client
import anthropic

load_dotenv()

# --- Spotify track ownership verification ---
_spotify_token_cache = {'token': None, 'expires': 0}

def _get_spotify_token():
    """Get a cached Spotify client credentials token."""
    if _spotify_token_cache['token'] and time.time() < _spotify_token_cache['expires']:
        return _spotify_token_cache['token']
    try:
        # Use the working Spotify app creds (sonicconverter-web app)
        resp = _requests.post('https://accounts.spotify.com/api/token', data={
            'grant_type': 'client_credentials',
            'client_id': '0891a3a7794c421f9121f77863df7b48',
            'client_secret': '561e769c8fbc4a4cb3d26884dc6163ed',
        })
        data = resp.json()
        _spotify_token_cache['token'] = data.get('access_token')
        _spotify_token_cache['expires'] = time.time() + data.get('expires_in', 3600) - 60
        return _spotify_token_cache['token']
    except Exception:
        return None

def _verify_track_owner(isrc, sp_artist_id):
    """Check if ISRC belongs to sp_artist_id on Spotify. Returns True if match or can't verify."""
    if not isrc or not sp_artist_id:
        return True
    try:
        token = _get_spotify_token()
        if not token:
            return True
        resp = _requests.get(f'https://api.spotify.com/v1/search?q=isrc:{isrc}&type=track&limit=1',
                            headers={'Authorization': f'Bearer {token}'})
        items = resp.json().get('tracks', {}).get('items', [])
        if not items:
            return True  # Track not found on Spotify, allow it
        track_artist_ids = [a['id'] for a in items[0].get('artists', [])]
        return sp_artist_id in track_artist_ids
    except Exception:
        return True  # On error, allow it

COUNTRY_CODES = {
    'US': 'United States', 'UK': 'United Kingdom', 'GB': 'United Kingdom',
    'FR': 'France', 'DE': 'Germany', 'BR': 'Brazil', 'IT': 'Italy',
    'ES': 'Spain', 'MX': 'Mexico', 'JP': 'Japan', 'KR': 'Korea',
    'AU': 'Australia', 'CA': 'Canada', 'NL': 'Netherlands', 'SE': 'Sweden',
    'NO': 'Norway', 'DK': 'Denmark', 'FI': 'Finland', 'BE': 'Belgium',
    'AT': 'Austria', 'CH': 'Switzerland', 'PL': 'Poland', 'PT': 'Portugal',
    'IE': 'Ireland', 'NZ': 'New Zealand', 'AR': 'Argentina', 'CL': 'Chile',
    'CO': 'Colombia', 'PE': 'Peru', 'ZA': 'South Africa', 'IN': 'India',
    'PH': 'Philippines', 'ID': 'Indonesia', 'TH': 'Thailand', 'VN': 'Vietnam',
    'MY': 'Malaysia', 'SG': 'Singapore', 'TW': 'Taiwan', 'HK': 'Hong Kong',
    'CN': 'China', 'RU': 'Russia', 'UA': 'Ukraine', 'TR': 'Turkey',
    'GR': 'Greece', 'CZ': 'Czech Republic', 'HU': 'Hungary', 'RO': 'Romania',
    'IL': 'Israel', 'EG': 'Egypt', 'NG': 'Nigeria', 'KE': 'Kenya',
    'PR': 'Puerto Rico', 'DO': 'Dominican Republic', 'VE': 'Venezuela',
    'EC': 'Ecuador', 'CR': 'Costa Rica', 'PA': 'Panama', 'GT': 'Guatemala',
    'CU': 'Cuba', 'JM': 'Jamaica', 'TT': 'Trinidad and Tobago',
    'IS': 'Iceland', 'LU': 'Luxembourg', 'SK': 'Slovakia', 'SI': 'Slovenia',
    'HR': 'Croatia', 'RS': 'Serbia', 'BG': 'Bulgaria', 'LT': 'Lithuania',
    'LV': 'Latvia', 'EE': 'Estonia',
    'KW': 'Kuwait', 'AE': 'United Arab Emirates', 'SA': 'Saudi Arabia',
    'QA': 'Qatar', 'BH': 'Bahrain', 'OM': 'Oman', 'JO': 'Jordan',
    'LB': 'Lebanon', 'IQ': 'Iraq', 'MA': 'Morocco', 'TN': 'Tunisia',
    'GH': 'Ghana', 'TZ': 'Tanzania', 'UG': 'Uganda', 'SN': 'Senegal',
    'BD': 'Bangladesh', 'PK': 'Pakistan', 'LK': 'Sri Lanka', 'NP': 'Nepal',
    'MM': 'Myanmar', 'KH': 'Cambodia', 'LA': 'Laos',
    'UY': 'Uruguay', 'PY': 'Paraguay', 'BO': 'Bolivia', 'HN': 'Honduras',
    'SV': 'El Salvador', 'NI': 'Nicaragua', 'MT': 'Malta', 'CY': 'Cyprus',
    'BA': 'Bosnia and Herzegovina', 'ME': 'Montenegro', 'MK': 'North Macedonia',
    'AL': 'Albania', 'MD': 'Moldova', 'GE': 'Georgia', 'AM': 'Armenia',
    'AZ': 'Azerbaijan', 'KZ': 'Kazakhstan', 'UZ': 'Uzbekistan',
}

def get_country_name(code):
    """Convert 2-letter country code to full name."""
    if not code:
        return 'N/A'
    return COUNTRY_CODES.get(code.upper(), code)


DEMONYMS = {
    'Germany': ['German', 'german'],
    'France': ['French', 'french'],
    'Brazil': ['Brazilian', 'brazilian'],
    'Italy': ['Italian', 'italian'],
    'Spain': ['Spanish', 'spanish'],
    'Mexico': ['Mexican', 'mexican'],
    'Japan': ['Japanese', 'japanese'],
    'Korea': ['Korean', 'korean'],
    'Argentina': ['Argentine', 'argentine', 'Argentinian', 'argentinian'],
    'Netherlands': ['Dutch', 'dutch'],
    'Sweden': ['Swedish', 'swedish'],
    'Norway': ['Norwegian', 'norwegian'],
    'Denmark': ['Danish', 'danish'],
    'Indonesia': ['Indonesian', 'indonesian'],
    'Portugal': ['Portuguese', 'portuguese'],
    'Poland': ['Polish', 'polish'],
    'Russia': ['Russian', 'russian'],
    'China': ['Chinese', 'chinese'],
    'Thailand': ['Thai', 'thai'],
    'Finland': ['Finnish', 'finnish'],
    'Austria': ['Austrian', 'austrian'],
    'Belgium': ['Belgian', 'belgian'],
    'Switzerland': ['Swiss', 'swiss'],
    'Greece': ['Greek', 'greek'],
    'Turkey': ['Turkish', 'turkish'],
    'Colombia': ['Colombian', 'colombian'],
    'Chile': ['Chilean', 'chilean'],
    'Peru': ['Peruvian', 'peruvian'],
}


def strip_demonyms(text):
    """Remove redundant demonyms when country is mentioned."""
    if not text:
        return text
    result = text
    for country, demonyms in DEMONYMS.items():
        if f'out of {country}' in result:
            for dem in demonyms:
                result = result.replace(f'the {dem} ', 'the ')
                # Also strip bare demonym before genre/descriptor (e.g. "Indonesian hip-hop out of Indonesia" -> "hip-hop out of Indonesia")
                result = result.replace(f'{dem} ', '', 1)
    return result

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL_CREATIVE = "claude-sonnet-4-20250514"
MODEL_STRUCTURED = "claude-haiku-4-5-20251001"

# ── Email recipient detection ──────────────────────────────────

MANAGER_DOMAIN_KEYWORDS = [
    'management', 'mgmt', 'records', 'label', 'entertainment', 'agency',
    'publicity', 'pr-', '-pr', 'media', 'booking', 'promotions', 'artists',
    'talent', 'creative', 'music', 'group', 'worldwide', 'global'
]

GENERIC_EMAIL_PREFIXES = [
    'info', 'contact', 'hello', 'hi', 'booking', 'bookings', 'music',
    'band', 'management', 'mgmt', 'press', 'pr', 'media', 'admin',
    # Foreign-language equivalents
    'post', 'kontakt', 'hei', 'hola', 'ciao', 'salut', 'hallo',
    'office', 'staff', 'support', 'team', 'hot', 'core',
    'contratacion', 'managers', 'agentur', 'bureau',
]

# Common first names to detect manager emails
COMMON_FIRST_NAMES = {
    # A
    'aaron', 'abbey', 'abby', 'abigail', 'ada', 'adam', 'adele', 'adelaine', 'adrian', 'adriana',
    'adrienne', 'agnes', 'aidan', 'aiden', 'aimee', 'aisha', 'al', 'alan', 'albert', 'alberto',
    'alejandro', 'alejo', 'alex', 'alexander', 'alexandra', 'alexis', 'alfonso', 'alfred', 'ali',
    'alice', 'alicia', 'alina', 'alison', 'allison', 'alma', 'alvaro', 'alvin', 'alyssa', 'amanda',
    'amber', 'amelia', 'amy', 'ana', 'andre', 'andrea', 'andres', 'andrew', 'andy', 'angel',
    'angela', 'angelica', 'angelina', 'angie', 'anita', 'ann', 'anna', 'anne', 'annette', 'anthony',
    'antonio', 'april', 'archie', 'ariana', 'arianna', 'ariel', 'arlene', 'armando', 'arnold',
    'arthur', 'arturo', 'ashley', 'asher', 'astrid', 'aubrey', 'audrey', 'august', 'augustine',
    'aurora', 'austin', 'autumn', 'ava', 'avery',
    # B
    'bailey', 'barbara', 'barry', 'beatrice', 'beatriz', 'becky', 'belinda', 'bella', 'ben',
    'benedict', 'benjamin', 'bennett', 'benny', 'bernadette', 'bernard', 'bernice', 'bertha',
    'beth', 'bethany', 'betty', 'beverly', 'bianca', 'bill', 'billy', 'bj', 'blake', 'blanche',
    'bo', 'bob', 'bobby', 'bonnie', 'brad', 'bradley', 'brandon', 'brandy', 'brayden', 'brenda',
    'brendan', 'brennan', 'brent', 'brett', 'brian', 'brianna', 'bridget', 'brittany', 'brittney',
    'brock', 'brodie', 'brooke', 'brooklyn', 'bruce', 'bruno', 'bryan', 'bryant', 'bryce', 'byron',
    # C
    'caitlin', 'caleb', 'calvin', 'cameron', 'camila', 'camille', 'candace', 'candice', 'cara',
    'carl', 'carla', 'carlo', 'carlos', 'carmen', 'carol', 'carolina', 'caroline', 'carolyn',
    'carrie', 'carson', 'carter', 'casey', 'cassandra', 'cassidy', 'catherine', 'cecelia', 'cecilia',
    'cedric', 'celeste', 'celia', 'cesar', 'chad', 'chance', 'chandler', 'charlene', 'charles',
    'charlie', 'charlotte', 'chase', 'chelsea', 'cheri', 'cheryl', 'chester', 'chloe', 'chris',
    'christian', 'christina', 'christine', 'christopher', 'cindy', 'claire', 'clara', 'clare',
    'clarence', 'clark', 'claude', 'claudia', 'clay', 'clayton', 'cliff', 'clifford', 'clifton',
    'clint', 'clinton', 'clyde', 'cody', 'cohen', 'colby', 'cole', 'colette', 'colin', 'colleen',
    'collin', 'colton', 'connie', 'connor', 'conrad', 'constance', 'cora', 'corey', 'corinne',
    'cornelius', 'cortney', 'cory', 'courtney', 'craig', 'cristina', 'cruz', 'crystal', 'curtis',
    'cynthia', 'cyrus',
    # D
    'daisy', 'dakota', 'dale', 'dallas', 'dalton', 'damian', 'damien', 'damon', 'dan', 'dana',
    'dane', 'dani', 'daniel', 'daniela', 'danielle', 'danny', 'dante', 'daphne', 'darius',
    'darlene', 'darnell', 'darrell', 'darren', 'darryl', 'darwin', 'dave', 'david', 'dawn',
    'dean', 'deanna', 'debbie', 'deborah', 'debra', 'dee', 'deidre', 'delilah', 'della', 'denise',
    'dennis', 'derek', 'derrick', 'desiree', 'desmond', 'destiny', 'devin', 'devon', 'dexter',
    'diana', 'diane', 'dianne', 'dick', 'diego', 'dillon', 'dina', 'dion', 'dixie', 'dolores',
    'domingo', 'dominic', 'dominick', 'dominik', 'dominique', 'don', 'donald', 'donna', 'donnie',
    'donovan', 'dora', 'doreen', 'doris', 'dorothy', 'doug', 'douglas', 'drake', 'drew', 'duane',
    'duncan', 'dustin', 'dwight', 'dylan',
    # E
    'earl', 'earnest', 'ed', 'eddie', 'edgar', 'edith', 'edmond', 'edmund', 'edna', 'eduardo',
    'edward', 'edwin', 'eileen', 'elaine', 'eleanor', 'elena', 'eli', 'elias', 'elijah', 'elisa',
    'elise', 'eliza', 'elizabeth', 'ella', 'ellen', 'ellie', 'elliot', 'elliott', 'ellis', 'elmer',
    'eloise', 'elsa', 'elsie', 'elton', 'elvis', 'emanuel', 'emerson', 'emilia', 'emilio', 'emily',
    'emma', 'emmanuel', 'emmett', 'enrique', 'eric', 'erica', 'erick', 'erik', 'erika', 'erin',
    'ernest', 'ernesto', 'ernie', 'erwin', 'esperanza', 'esteban', 'estelle', 'esther', 'ethan',
    'ethel', 'eugene', 'eunice', 'eva', 'evan', 'evangeline', 'eve', 'evelyn', 'everett',
    'ezekiel', 'ezra',
    # F
    'fabian', 'faith', 'fatima', 'faye', 'federico', 'felicia', 'felipe', 'felix', 'fern',
    'fernanda', 'fernando', 'fletcher', 'flor', 'flora', 'florence', 'floyd', 'frances',
    'francesca', 'francine', 'francis', 'francisco', 'frank', 'frankie', 'franklin', 'fred',
    'freddie', 'frederick', 'freya', 'fritz',
    # G
    'gabriel', 'gabriela', 'gabriella', 'gabrielle', 'gail', 'gale', 'gareth', 'garrett', 'garry',
    'gary', 'gavin', 'gayle', 'gene', 'genevieve', 'geoffrey', 'george', 'georgia', 'georgina',
    'gerald', 'geraldine', 'gerard', 'gerardo', 'gero', 'gerry', 'gertrude', 'gigi', 'gilbert',
    'gina', 'ginger', 'giovanni', 'gladys', 'glen', 'glenn', 'gloria', 'gonzalo', 'gordon',
    'grace', 'grady', 'graham', 'grant', 'greg', 'gregory', 'greta', 'gretchen', 'guillermo',
    'gus', 'gustavo', 'gwen', 'gwendolyn',
    # H
    'hailey', 'haley', 'hana', 'hank', 'hanna', 'hannah', 'hans', 'harlan', 'harley', 'harold',
    'harper', 'harriet', 'harris', 'harrison', 'harry', 'harvey', 'hayden', 'hazel', 'heath',
    'heather', 'hector', 'heidi', 'helen', 'helena', 'helene', 'helga', 'henry', 'herb', 'herbert',
    'herman', 'hilda', 'hillary', 'holly', 'homer', 'hope', 'horace', 'howard', 'hubert', 'huey',
    'hugh', 'hugo', 'humberto', 'hunter',
    # I
    'ian', 'ida', 'ignacio', 'ike', 'ilene', 'imogene', 'inbal', 'inez', 'ingrid', 'ira', 'irene',
    'iris', 'irma', 'irving', 'irwin', 'isaac', 'isabel', 'isabella', 'isaiah', 'isidro', 'ismael',
    'israel', 'ivan', 'ivy',
    # J
    'jace', 'jack', 'jackie', 'jackson', 'jacob', 'jacqueline', 'jade', 'jaden', 'jaime', 'jake',
    'jamal', 'james', 'jamie', 'jamil', 'jan', 'jana', 'jane', 'janelle', 'janet', 'janette',
    'janice', 'janine', 'janis', 'jared', 'jarrett', 'jasmine', 'jason', 'jasper', 'javier', 'jay',
    'jayden', 'jayson', 'jean', 'jeanette', 'jeanne', 'jeannette', 'jeannie', 'jed', 'jeff',
    'jefferson', 'jeffery', 'jeffrey', 'jelle', 'jen', 'jenna', 'jennie', 'jennifer', 'jenny',
    'jeremiah', 'jeremy', 'jermaine', 'jerome', 'jerry', 'jess', 'jesse', 'jessica', 'jessie',
    'jesus', 'jet', 'jewel', 'jhon', 'jill', 'jillian', 'jim', 'jimmie', 'jimmy', 'jo', 'joan',
    'joann', 'joanna', 'joanne', 'joaquin', 'jocelyn', 'jodi', 'jodie', 'jody', 'joe', 'joel',
    'joey', 'johanna', 'john', 'johnathan', 'johnnie', 'johnny', 'jon', 'jonah', 'jonas',
    'jonathan', 'jonathon', 'jordan', 'jorge', 'jose', 'joseph', 'josephine', 'josh', 'joshua',
    'josiah', 'josie', 'josue', 'joy', 'joyce', 'juan', 'juanita', 'jude', 'judith', 'judy',
    'julia', 'julian', 'juliana', 'julianna', 'julianne', 'julie', 'juliet', 'juliette', 'julio', 'julius',
    'june', 'junior', 'justin', 'justine',
    # K
    'kai', 'kaitlyn', 'kara', 'karen', 'karina', 'karl', 'karla', 'kasey', 'kate', 'katelyn',
    'katherine', 'kathleen', 'kathryn', 'kathy', 'katie', 'katrina', 'kay', 'kayla', 'kaylee',
    'keegan', 'keenan', 'keith', 'kelley', 'kelly', 'kelsey', 'kelvin', 'ken', 'kendall', 'kendra',
    'kendrick', 'kenneth', 'kenny', 'kent', 'kerry', 'keven', 'kevin', 'khloe', 'kieran', 'kim',
    'kimberly', 'king', 'kirby', 'kirk', 'kirsten', 'kit', 'korey', 'kris', 'krista', 'kristen',
    'kristi', 'kristin', 'kristina', 'kristine', 'kristy', 'krystal', 'kurt', 'kurtis', 'kyle',
    'kylie', 'kyra',
    # L
    'lacey', 'lamar', 'lambo', 'lana', 'lance', 'landon', 'lane', 'lara', 'larissa', 'larry',
    'lars', 'lasse', 'latasha', 'latoya', 'laura', 'laurel', 'lauren', 'laurence', 'laurie',
    'laverne', 'lawrence', 'layla', 'lea', 'leah', 'leandro', 'leann', 'leanna', 'lee', 'leigh',
    'leira', 'lena', 'lenore', 'leo', 'leon', 'leona', 'leonard', 'leonardo', 'leonel', 'leopoldo',
    'leroy', 'les', 'lesley', 'leslie', 'lester', 'leticia', 'levi', 'lewis', 'liam', 'lila',
    'lilian', 'lillian', 'lillie', 'lily', 'lina', 'lincoln', 'linda', 'lindsay', 'lindsey',
    'lionel', 'lisa', 'liz', 'lloyd', 'logan', 'lois', 'lola', 'lon', 'lonnie', 'lora', 'lorena',
    'lorenzo', 'loretta', 'lori', 'lorraine', 'lottie', 'lou', 'louie', 'louis', 'louisa',
    'louise', 'lourdes', 'luana', 'luca', 'lucas', 'lucia', 'luciana', 'lucille', 'lucinda',
    'lucy', 'luis', 'luisa', 'luke', 'luna', 'luther', 'luz', 'lydia', 'lynda', 'lynette', 'lynn',
    # M
    'mabel', 'mack', 'mackenzie', 'macartney', 'macy', 'madeleine', 'madeline', 'madison', 'mae',
    'maggie', 'mahalia', 'maisie', 'malcolm', 'malik', 'mallory', 'mandy', 'manuel', 'mara',
    'marc', 'marcel', 'marcella', 'marcelo', 'marcia', 'marcio', 'marco', 'marcos', 'marcus',
    'margaret', 'margarita', 'margot', 'marguerite', 'maria', 'mariah', 'marian', 'mariana',
    'marianne', 'marie', 'marilyn', 'marina', 'mario', 'marion', 'marisa', 'marisol', 'marissa',
    'marjorie', 'mark', 'marlene', 'marlon', 'marquis', 'marshall', 'marta', 'martha', 'martin',
    'martina', 'marty', 'marvin', 'mary', 'maryann', 'mason', 'mathew', 'mathilde', 'matt',
    'matthew', 'mateo', 'matias', 'maude', 'maureen', 'maurice', 'mauricio', 'mavis', 'max',
    'maxine', 'maxwell', 'maya', 'meagan', 'megan', 'meghan', 'melanie', 'melba', 'melinda',
    'melissa', 'melodie', 'melody', 'melvin', 'mercedes', 'meredith', 'merle', 'meryl', 'mia',
    'micah', 'michael', 'michaela', 'michele', 'michelle', 'mickey', 'miguel', 'mike', 'mila',
    'milan', 'mildred', 'miles', 'millie', 'milo', 'milton', 'mindy', 'minerva', 'minnie',
    'miranda', 'miriam', 'misty', 'mitch', 'mitchell', 'molly', 'mona', 'monica', 'monique',
    'monroe', 'monte', 'monty', 'morgan', 'morris', 'moses', 'muriel', 'murray', 'myles', 'myra',
    'myrna', 'myrtle',
    # N
    'nadia', 'nadine', 'nancy', 'naomi', 'natalia', 'natalie', 'natasha', 'nate', 'nathalie',
    'nathan', 'nathaniel', 'neal', 'ned', 'neil', 'nellie', 'nelson', 'nestor', 'nicholas', 'nick',
    'nickolas', 'nicky', 'nicolas', 'nicole', 'nigel', 'nina', 'noah', 'noel', 'noelle', 'nola',
    'nolan', 'nora', 'norma', 'norman', 'nuria',
    # O
    'octavia', 'octavio', 'odell', 'olga', 'olive', 'oliver', 'olivia', 'omar', 'opal', 'ophelia',
    'ora', 'orlando', 'oscar', 'otis', 'otto', 'owen',
    # P
    'pablo', 'paddy', 'paige', 'paloma', 'pamela', 'paris', 'parker', 'pat', 'patricia', 'patrick',
    'patsy', 'patti', 'patty', 'paul', 'paula', 'pauline', 'paulo', 'pearl', 'pedro', 'peggy',
    'penelope', 'penny', 'percy', 'perry', 'pete', 'peter', 'petrina', 'phil', 'philip', 'phillip',
    'phoebe', 'phyllis', 'pilar', 'pierre', 'polly', 'porter', 'preston', 'prince', 'priscilla',
    # Q
    'queen', 'quentin', 'quincy', 'quinn',
    # R
    'rachel', 'rafael', 'ralph', 'ramiro', 'ramon', 'ramona', 'randall', 'randolph', 'randy',
    'raoul', 'raphael', 'raquel', 'rashad', 'raul', 'raven', 'ray', 'raymond', 'reagan', 'rebecca',
    'reed', 'reggie', 'reginald', 'reid', 'rene', 'renata', 'renato', 'renee', 'rex', 'rhonda',
    'ricardo', 'rich', 'richard', 'richie', 'rick', 'ricky', 'rico', 'riley', 'rita', 'rob',
    'robbie', 'robert', 'roberta', 'roberto', 'robin', 'rocco', 'rochelle', 'rocio', 'rocky',
    'rod', 'roderick', 'rodney', 'rodolfo', 'rodrigo', 'rogelio', 'roger', 'roland', 'rolando',
    'roman', 'romeo', 'ron', 'ronald', 'ronnie', 'rory', 'rosa', 'rosalie', 'rosalind', 'rosalyn',
    'rosanna', 'rose', 'rosemary', 'rosetta', 'rosie', 'ross', 'rowan', 'roxanne', 'roy', 'royce',
    'ruben', 'ruby', 'rudolph', 'rudy', 'rufus', 'rupert', 'russell', 'rusty', 'ruth', 'ryan',
    # S
    'sabrina', 'sadie', 'sal', 'sally', 'salvador', 'salvatore', 'sam', 'samantha', 'sammie',
    'sammy', 'samuel', 'sandra', 'sandy', 'santiago', 'santos', 'sara', 'sarah', 'saul', 'savannah',
    'scott', 'sean', 'sebastian', 'selena', 'serena', 'sergio', 'seth', 'seymour', 'shane',
    'shannon', 'shari', 'sharna', 'sharon', 'shaun', 'shauna', 'shawn', 'sheila', 'shelby',
    'sheldon', 'shelley', 'shelly', 'sheri', 'sherri', 'sheryl', 'shirley', 'sid', 'sidney',
    'sierra', 'silvia', 'simon', 'simone', 'silas', 'sofia', 'solomon', 'sondra', 'sonia', 'sonja',
    'sonya', 'sophia', 'sophie', 'spencer', 'stacey', 'staci', 'stacy', 'stan', 'stanford',
    'stanley', 'stefan', 'stella', 'stephan', 'stephanie', 'stephen', 'sterling', 'steve', 'steven',
    'stewart', 'stuart', 'sue', 'sullivan', 'summer', 'susan', 'susanna', 'susanne', 'susie',
    'suzanne', 'suzette', 'sybil', 'sydney', 'sylvia',
    # T
    'tabitha', 'tamara', 'tami', 'tamika', 'tammy', 'tanya', 'tara', 'taryn', 'tatiana', 'taylor',
    'ted', 'teddy', 'terence', 'teresa', 'teri', 'terra', 'terrance', 'terrell', 'terrence',
    'terri', 'terry', 'tessa', 'thad', 'thea', 'thelma', 'theo', 'theodore', 'theresa', 'therese',
    'thomas', 'thurman', 'tiago', 'tiffany', 'tim', 'timmy', 'timothy', 'tina', 'titus', 'tobias',
    'toby', 'todd', 'tom', 'tomas', 'tommie', 'tommy', 'toni', 'tony', 'tonya', 'tracey', 'traci',
    'tracy', 'travis', 'trent', 'trenton', 'trevor', 'trey', 'tricia', 'trina', 'trisha',
    'tristan', 'troy', 'trudy', 'tucker', 'ty', 'tyler', 'tyrone', 'tyson',
    # U
    'ulysses', 'ursula',
    # V
    'val', 'valentina', 'valeria', 'valerie', 'van', 'vance', 'vanessa', 'vaughn', 'velma', 'vera',
    'vern', 'vernon', 'veronica', 'vicente', 'vicki', 'vicky', 'victor', 'victoria', 'vince',
    'vincent', 'vinicius', 'viola', 'violet', 'virgil', 'virginia', 'vivian', 'viviana',
    # W
    'wade', 'walker', 'wallace', 'wally', 'walter', 'wanda', 'ward', 'warner', 'warren', 'wayne',
    'wendell', 'wendy', 'wes', 'wesley', 'weston', 'whitney', 'wilbert', 'wilbur', 'wilfred',
    'will', 'willa', 'willard', 'william', 'willie', 'willis', 'wilma', 'wilson', 'winnie',
    'winifred', 'winston', 'woodrow', 'wyatt',
    # X
    'xavier', 'ximena',
    # Y
    'yolanda', 'yvette', 'yvonne',
    # Z
    'zachary', 'zane', 'zelda', 'zoe', 'zoey', 'zora',
    # French
    'alain', 'arnaud', 'aurelie', 'benoit', 'brigitte', 'camille', 'cedric', 'celine',
    'chantal', 'christophe', 'corinne', 'damien', 'delphine', 'dominique', 'eloise', 'emile',
    'etienne', 'fabien', 'fabienne', 'florian', 'francois', 'frederique', 'gaelle', 'gilles',
    'guillaume', 'helene', 'herve', 'isabelle', 'jacques', 'julien', 'laurent', 'loic', 'luc',
    'lucien', 'manon', 'margaux', 'marion', 'mathieu', 'maxime', 'monique', 'nathalie',
    'olivier', 'pascal', 'philippe', 'quentin', 'remi', 'renaud', 'romain', 'sandrine',
    'sebastien', 'severine', 'solange', 'sylvie', 'thierry', 'valentin', 'veronique',
    'virginie', 'yves',
    # German / Austrian / Swiss
    'achim', 'andreas', 'angelika', 'anja', 'axel', 'bernd', 'birgit', 'carsten', 'claus',
    'detlef', 'dieter', 'dirk', 'dorothea', 'elke', 'erika', 'ernst', 'franz', 'frederik',
    'gabi', 'gerhard', 'gisela', 'guenther', 'harald', 'heinrich', 'helmut', 'horst', 'ingo',
    'jens', 'joerg', 'juergen', 'jurgen', 'katja', 'klaus', 'lukas', 'manfred', 'markus',
    'monika', 'norbert', 'olaf', 'petra', 'rainer', 'ralf', 'reinhard', 'renate', 'rolf',
    'sabine', 'sascha', 'sigrid', 'silke', 'steffen', 'susanne', 'sven', 'thorsten', 'tobias',
    'uwe', 'ulf', 'volker', 'werner', 'wolfgang',
    # Scandinavian
    'anders', 'anette', 'annika', 'arne', 'birgitta', 'bjorn', 'dag', 'einar', 'elisabet',
    'frida', 'geir', 'gunnar', 'gustav', 'henrik', 'ida', 'ingrid', 'johan', 'jona', 'jorgen',
    'karin', 'katarina', 'knut', 'kristin', 'leif', 'linnea', 'lotta', 'magnus', 'maja', 'malin',
    'mats', 'mette', 'mikael', 'morten', 'nils', 'odd', 'ola', 'oskar', 'per', 'petter',
    'ragnhild', 'roar', 'signe', 'sigurd', 'siri', 'solveig', 'stein', 'stian', 'stine', 'svein',
    'terje', 'tor', 'torbjorn', 'tore', 'trond', 'vibeke', 'viggo',
    # Dutch / Belgian
    'arjan', 'bas', 'bram', 'daan', 'dries', 'edwin', 'els', 'femke', 'geert', 'gerben', 'gert',
    'guido', 'harm', 'henk', 'inge', 'jaap', 'jeroen', 'joost', 'kees', 'lotte', 'maarten',
    'maartje', 'marieke', 'martijn', 'matthijs', 'niels', 'pieter', 'pim', 'rens', 'rik', 'ruud',
    'sander', 'sjors', 'thijs', 'tijs', 'wim', 'wouter',
    # Italian
    'adriano', 'aldo', 'alessandra', 'alessandro', 'alfredo', 'angelo', 'antonella', 'benedetta',
    'chiara', 'claudio', 'dario', 'davide', 'eleonora', 'enrico', 'fabrizio', 'federica',
    'filippo', 'francesca', 'franco', 'giacomo', 'gianluca', 'gianmarco', 'giorgio', 'giovanna',
    'giulia', 'giuliano', 'giuseppe', 'ilaria', 'lara', 'luciano', 'luigi', 'margherita',
    'massimo', 'matteo', 'maurizio', 'nicola', 'paola', 'paolo', 'piero', 'raffaele', 'riccardo',
    'salvatore', 'stefano', 'vincenzo', 'vittorio',
    # Spanish / Portuguese (not already in A-Z above)
    'alejo', 'alvaro', 'caio', 'concha', 'diogo', 'fabio', 'filipe', 'flavio', 'ines', 'joao',
    'leira', 'lorena', 'luana', 'luciana', 'marcio', 'margarita', 'marisa', 'natalia',
    'pilar', 'rocio', 'santiago', 'tomas', 'valeria',
    # Korean / Japanese (romanized)
    'akira', 'daiki', 'haruki', 'hiroshi', 'kenji', 'koji', 'makoto', 'naoki', 'ryo', 'satoshi',
    'shin', 'shoji', 'takeshi', 'taro', 'yuki', 'yuko', 'bora', 'daesung', 'eunji', 'hyun',
    'jimin', 'jisoo', 'joon', 'minji', 'minho', 'seung', 'soojin', 'sunho', 'woojin', 'yuna',
    # Eastern European
    'aleksander', 'andrei', 'boris', 'catalin', 'damir', 'dimitri', 'dmitri', 'dragana', 'dusan',
    'filip', 'jana', 'kirill', 'lazar', 'ludmila', 'marko', 'miroslav', 'nikola', 'oleg',
    'radek', 'sergei', 'stanislav', 'svetlana', 'tatiana', 'vladimir', 'yuri', 'zoran',
    # Turkish / Middle Eastern
    'ahmet', 'aylin', 'berk', 'burak', 'cem', 'deniz', 'elif', 'emir', 'fatih', 'gamze',
    'hakan', 'kemal', 'leyla', 'mehmet', 'murat', 'mustafa', 'nizar', 'onur', 'rami', 'selim',
    'selin', 'tariq', 'yasmin', 'yusuf', 'zeynep',
}


def shorten_artist_name(name):
    """Shorten to first name if it looks like a person's name. Keep band names full."""
    if not name:
        return 'there'
    parts = name.strip().split()
    if len(parts) == 1:
        return name  # Mononym or band name
    if len(parts) >= 4:
        return name  # 4+ words = likely band/project
    lower = name.lower()
    band_signals = [' the ', ' & ', ' and ', ' vs ', ' vs. ', ' duo', ' band', ' collective', ' orchestra', ' choir']
    if lower.startswith('the ') or any(sig in lower for sig in band_signals):
        return name
    first_lower = parts[0].lower()
    # Words that look like names but are actually common artist prefixes — never shorten to these
    artist_prefixes = {
        'lil', 'lil\'', 'big', 'young', 'old', 'baby', 'little', 'la', 'el', 'le', 'los', 'las',
        'dj', 'mc', 'dr', 'mr', 'mrs', 'ms', 'sir', 'king', 'queen', 'queenz', 'prince',
        'princess', 'lord', 'lady', 'saint', 'st', 'bishop', 'father', 'brother', 'sister',
        'captain', 'major', 'general', 'chief', 'divine', 'royal', 'super', 'mega',
        'kid', 'boy', 'girl', 'slim', 'fat', 'lil\'', 'black', 'white', 'red', 'blue',
    }
    if first_lower in artist_prefixes:
        return name

    # 2-3 word name where first word is a known first name → use first name only
    if first_lower in COMMON_FIRST_NAMES:
        return parts[0]
    # Strip trailing letter variant: "Krystall" → "krystal" which IS in the set
    if len(first_lower) >= 4 and first_lower[:-1] in COMMON_FIRST_NAMES:
        return parts[0]
    # LLM fallback — ask Haiku if the first word is a person's first name
    try:
        response = claude.messages.create(
            model=MODEL_STRUCTURED,
            max_tokens=20,
            messages=[{"role": "user", "content": f'Is "{parts[0]}" a person\'s first name (not a title, prefix, or stage name word)? Reply ONLY "yes" or "no". Examples: "Julianna" → "yes", "Doomtree" → "no", "Robbie" → "yes", "SPIKY" → "no", "Luiza" → "yes", "Teen" → "no", "Young" → "no", "Lil" → "no", "Bishop" → "no", "Divine" → "no"'}],
        )
        answer = response.content[0].text.strip().lower()
        if answer == 'yes':
            return parts[0]
    except Exception:
        pass
    return name


def normalize_name(name):
    """Normalize a name for comparison - lowercase, remove special chars."""
    if not name:
        return ''
    import re
    # Remove special chars, keep alphanumeric
    return re.sub(r'[^a-z0-9]', '', name.lower())


def detect_email_recipient(email, artist_name):
    """
    Detect if email is addressed to the artist or a manager/rep.

    Returns:
        {"type": "artist"} - email goes to artist directly
        {"type": "manager", "name": "Andrea"} - email goes to a rep
    """
    if not email or not artist_name:
        return {"type": "artist"}

    email = email.lower().strip()
    artist_normalized = normalize_name(artist_name)

    # Split email into local part and domain
    if '@' not in email:
        return {"type": "artist"}

    local_part, domain = email.split('@', 1)
    domain_base = domain.split('.')[0]  # e.g., "riverside" from "riverside-mgmt.com"

    # Check if domain contains artist name (likely artist's own domain)
    if artist_normalized and len(artist_normalized) >= 3:
        if artist_normalized in normalize_name(domain):
            return {"type": "artist"}

    # Check if local part contains artist name
    if artist_normalized and len(artist_normalized) >= 3:
        if artist_normalized in normalize_name(local_part):
            return {"type": "artist"}

    # Check for generic prefixes (info@, contact@, etc.) - could be either
    # If domain doesn't match artist, probably a label/mgmt
    local_base = local_part.split('.')[0]  # handle firstname.lastname

    if local_base in GENERIC_EMAIL_PREFIXES:
        # Generic prefix at non-artist domain = likely label/mgmt, address generically
        # But we don't have a name, so treat as artist
        return {"type": "artist"}

    # Check if domain has manager/label keywords
    is_manager_domain = any(kw in domain.lower() for kw in MANAGER_DOMAIN_KEYWORDS)

    # Check if local part is a common first name
    # Handle formats: andrea@, andrea.smith@, asmith@
    potential_first_name = local_base

    # But first: if the email prefix matches the artist's own first name, it's the artist
    # e.g. "Andrea Balency" with andrea.mgmt@..., "Julianna Riolino" with julianna@gmail.com
    artist_parts = artist_name.strip().split()
    if len(artist_parts) >= 2:
        artist_first_lower = artist_parts[0].lower()
        if potential_first_name == artist_first_lower:
            return {"type": "artist"}
        # Also check firstname.lastname format: andrea.smith@ for "Andrea Balency"
        if '.' in local_part:
            first_part = local_part.split('.')[0]
            if first_part == artist_first_lower:
                return {"type": "artist"}

    if potential_first_name in COMMON_FIRST_NAMES:
        capitalized_name = potential_first_name.capitalize()
        return {"type": "manager", "name": capitalized_name}

    # Check firstname.lastname format
    if '.' in local_part:
        first_part = local_part.split('.')[0]
        if first_part in COMMON_FIRST_NAMES:
            capitalized_name = first_part.capitalize()
            return {"type": "manager", "name": capitalized_name}

    # Strip trailing single letter (mattb → matt, anthonyd → anthony)
    if len(potential_first_name) >= 4 and potential_first_name[:-1] in COMMON_FIRST_NAMES:
        capitalized_name = potential_first_name[:-1].capitalize()
        return {"type": "manager", "name": capitalized_name}

    # LLM fallback — ask Haiku if the local part looks like a person's name
    # Fires for ALL non-artist emails, not just manager domains
    if len(potential_first_name) >= 2:
        try:
            response = claude.messages.create(
                model=MODEL_STRUCTURED,
                max_tokens=50,
                messages=[{"role": "user", "content": f'Is "{local_part}" a person\'s first name or contains one? Reply ONLY with the first name capitalized, or "no". Examples: "mattb" → "Matt", "jobsgreenmusichk" → "no", "tsangiacomo" → "no", "jj" → "JJ", "dc" → "no", "promo" → "no", "sandy" → "Sandy", "bled" → "no", "g" → "no"'}],
            )
            answer = response.content[0].text.strip().strip('"').strip("'")
            if answer.lower() != 'no' and len(answer) >= 2 and len(answer) <= 20:
                return {"type": "manager", "name": answer}
        except Exception:
            pass

    # If manager domain but no identifiable name, still treat as manager with generic greeting
    if is_manager_domain:
        return {"type": "manager", "name": None}

    # Default to artist
    return {"type": "artist"}


# ── Emotion taxonomy & compound matrix ──────────────────────────

EMOTION_TAXONOMY = """
Wonder: Wondrous, Awestruck, Moved, Piercing, Spellbound
Transcendence: Inspired, Celebratory, Euphoric, Spiritual, Thrilled
Tenderness: Tenderness, Sensual, Amorous, Affectionate, Soft
Nostalgia: Nostalgia, Dreamy, Sentimental, Longing, Melancholy
Peacefulness: Serene, Calm, Soothing, Relaxed, Meditative
Power: Powerful, Strong, Energetic, Triumphant, Fiery
Joyful: Joyful, Cheerful, Amused, Glad, Happy
Tension: Tense, Nervous, Frightened, Agitated, Shaken
Sadness: Sad, Sorrowful, Melancholy, Melancholic, Dejected
"""

EMOTION_TO_CATEGORY = {
    'Wondrous': 'Wonder', 'Awestruck': 'Wonder', 'Moved': 'Wonder',
    'Piercing': 'Wonder', 'Spellbound': 'Wonder', 'Wonder': 'Wonder',
    'Inspired': 'Transcendence', 'Celebratory': 'Transcendence', 'Euphoric': 'Transcendence',
    'Spiritual': 'Transcendence', 'Thrilled': 'Transcendence', 'Transcendence': 'Transcendence',
    'Tenderness': 'Tenderness', 'Sensual': 'Tenderness', 'Amorous': 'Tenderness',
    'Affectionate': 'Tenderness', 'Soft': 'Tenderness',
    'Nostalgia': 'Nostalgia', 'Dreamy': 'Nostalgia', 'Sentimental': 'Nostalgia',
    'Longing': 'Nostalgia', 'Melancholy': 'Nostalgia',
    'Serene': 'Peacefulness', 'Calm': 'Peacefulness', 'Soothing': 'Peacefulness',
    'Relaxed': 'Peacefulness', 'Meditative': 'Peacefulness', 'Peacefulness': 'Peacefulness',
    'Powerful': 'Power', 'Strong': 'Power', 'Energetic': 'Power',
    'Triumphant': 'Power', 'Fiery': 'Power', 'Power': 'Power',
    'Joyful': 'Joyful', 'Cheerful': 'Joyful', 'Amused': 'Joyful',
    'Glad': 'Joyful', 'Happy': 'Joyful',
    'Tense': 'Tension', 'Nervous': 'Tension', 'Frightened': 'Tension',
    'Agitated': 'Tension', 'Shaken': 'Tension', 'Tension': 'Tension',
    'Sad': 'Sadness', 'Sorrowful': 'Sadness', 'Melancholic': 'Sadness',
    'Dejected': 'Sadness', 'Sadness': 'Sadness',
}

COMPOUND_MATRIX = {
    'Wonder': {
        'Wonder': 'awe', 'Transcendence': 'bigger than the song', 'Tenderness': 'wonder but soft',
        'Nostalgia': 'curious and a little sad', 'Peacefulness': 'quietly fascinated', 'Power': 'jaw on the floor',
        'Joyful': 'joy and discovery at once', 'Tension': 'excited but nervous', 'Sadness': 'sad but gorgeous'
    },
    'Transcendence': {
        'Wonder': 'feels bigger than it should', 'Transcendence': 'leaves you somewhere else', 'Tenderness': 'lifts you gently',
        'Nostalgia': 'longing that takes you somewhere', 'Peacefulness': 'deep calm', 'Power': 'a release you can feel',
        'Joyful': 'elation', 'Tension': 'hope you can barely hold onto', 'Sadness': 'heavy but it lifts'
    },
    'Tenderness': {
        'Wonder': 'soft and surprising', 'Transcendence': 'tender but it takes you somewhere', 'Tenderness': 'affection',
        'Nostalgia': 'aches a little', 'Peacefulness': 'close and quiet', 'Power': 'soft but it hits',
        'Joyful': 'warm all the way through', 'Tension': 'love with an edge', 'Sadness': 'heartbreak but gentle'
    },
    'Nostalgia': {
        'Wonder': 'dreamy and far away', 'Transcendence': 'longing that pulls you up', 'Tenderness': 'warm memory',
        'Nostalgia': 'deep in the past', 'Peacefulness': 'looking back and being ok with it', 'Power': 'stubborn nostalgia',
        'Joyful': 'good memories', 'Tension': 'can not let go', 'Sadness': 'heavy with memory'
    },
    'Peacefulness': {
        'Wonder': 'still and wide open', 'Transcendence': 'clear headed', 'Tenderness': 'comfortable',
        'Nostalgia': 'settled', 'Peacefulness': 'total quiet', 'Power': 'calm but sure of itself',
        'Joyful': 'easy and content', 'Tension': 'peace that could break', 'Sadness': 'accepted sadness'
    },
    'Power': {
        'Wonder': 'raw and revelatory', 'Transcendence': 'triumphant', 'Tenderness': 'strong but gentle',
        'Nostalgia': 'holding onto something hard', 'Peacefulness': 'quiet power', 'Power': 'all force',
        'Joyful': 'victory lap energy', 'Tension': 'about to snap', 'Sadness': 'grief with teeth'
    },
    'Joyful': {
        'Wonder': 'wonder and fun at once', 'Transcendence': 'ecstatic', 'Tenderness': 'playful and sweet',
        'Nostalgia': 'happy looking back', 'Peacefulness': 'easy joy', 'Power': 'celebratory',
        'Joyful': 'happy', 'Tension': 'manic', 'Sadness': 'smiling through it'
    },
    'Tension': {
        'Wonder': 'can not look away', 'Transcendence': 'reaching for something', 'Tenderness': 'love that scares you',
        'Nostalgia': 'haunted', 'Peacefulness': 'barely holding together', 'Power': 'about to blow',
        'Joyful': 'frantic', 'Tension': 'at the edge', 'Sadness': 'desperate'
    },
    'Sadness': {
        'Wonder': 'tragic but beautiful', 'Transcendence': 'heavy but it means something', 'Tenderness': 'hurt and still loving',
        'Nostalgia': 'grieving', 'Peacefulness': 'quiet and empty', 'Power': 'sad and angry at the same time',
        'Joyful': 'crying happy', 'Tension': 'can not take much more', 'Sadness': 'deep in it'
    }
}

LYRIC_EMOTION_PROMPT = """Analyze these song lyrics and extract the emotions expressed.

EMOTION VOCABULARY (you MUST pick from these only):
{taxonomy}

LYRICS:
{lyrics}

TASK:
Read the lyrics and identify the 4 strongest emotions expressed. Pick ONLY from the vocabulary above.

Return ONLY valid JSON:
{{
  "lyric_emotion_1": "strongest emotion",
  "lyric_emotion_2": "second strongest",
  "lyric_emotion_3": "third strongest",
  "lyric_emotion_4": "fourth strongest"
}}"""

MIXER_OBSERVATION_PROMPT = """Generate 1-2 sentences about this track that sound like a mixer talking to another producer. Casual, specific, understated.

YOU MUST COVER ALL THREE:
1. SONIC CHARACTER — what does the mix sound like? Lead with whatever is MOST distinctive: could be brightness, midrange, low end, space, width, texture — don't default to bass every time.
2. EMOTIONAL FEEL — what does it make you feel? Use the emotion fields, translate to casual language.
3. THE TEXTURE — ONLY if something actually stands out. If how_squeezed, how_busy, or how_rough are way outside normal, mention it. If they're all normal, say NOTHING about texture. Most tracks are normal. Don't force a texture comment.

BIO BLURB (already sent to artist): {bio_blurb}
If the bio blurb mentions genre details, DO NOT repeat genre in your observation. Focus purely on sonic and emotional.
If the bio blurb is about an award, label, collab, origin, or location — you SHOULD reference genre in your observation BUT never repeat any country, city, or location name that already appears in the bio blurb.
If there is no bio blurb (null), include genre context.
If genre is "N/A" or unknown, do NOT guess or invent a genre — just describe the sonic and emotional qualities of what you hear.

GENRE NOTE: Use the artist's own genre tag as the primary genre. Reference artist genres are supplementary context only — they describe similar artists, not this artist. Never replace the artist's genre with a reference artist's genre. For example, if the artist is "death metal" but a reference artist is tagged "power metal," call it death metal.

INPUT:
- track: {track}
- genre: {genre}
- reference_artist_genres: {reference_genres}
- sonic_signature_text: {sonic_signature_text}
- tonal_balance_description: {tonal_balance_description}
- brightness_character: {brightness_character}
- bass_character: {bass_character}
- emotional_signature: {emotional_signature}
- emotion_1: {emotion_1}
- emotion_2: {emotion_2}
- genre_deviation_score: {genre_deviation_score}
- how_squeezed (0-1, normal is 0.80): {compression_amount}
- how_busy (0-1, normal is 0.07): {spectral_complexity}
- how_rough (0-1, normal is 0.15): {dissonance}

DEVIATION SCORE GUIDE:
- Above 0.6 = doing something different for the genre — worth a nod
- Below 0.6 = don't mention

CRITICAL — TRANSLATE, DON'T PARROT:
The emotion/tonal fields use analysis vocabulary. Convert to how a mixer actually talks:
- "ethereal" → "floaty", "airy", "spacey"
- "fat" → "thick", "full", "chunky"
- "warm" → "warm", "round", "smooth"
- "quirky" → "off-kilter", "weird in a good way", "unexpected"
- "piercing" → "bright and sharp", "cutting through", "right in your face"
- "dark" → "dark", "moody", "murky"
- "sub-heavy" → "lot of low end", "heavy bottom"
- "bass-heavy" → "bottom-heavy", "got weight down low"
- "balanced" → "sits right", "everything's in its lane"
- "transcendence" → "lifts", "opens up", "hits different"
- "powerful" → "hits hard", "punchy", "got weight"
- "tenderness" → "soft", "gentle", "delicate"
- "tension" → "edge", "grit", "something unsettled"
DON'T default to talking about bass. Lead with whatever the tonal_balance_description actually says.

TEXTURE — only if it's a real outlier:
- how_squeezed: >0.85 = "slammed", "crushed", "squeezed tight"; <0.75 = "open", "breathing room", "nothing's pushed"
- how_busy: >0.10 = "busy mix", "lot packed in there", "thick arrangement"; <0.05 = "sparse", "stripped back", "barely anything there"
- how_rough: >0.2 = "crunchy", "gritty", "got some dirt on it"; <0.05 = "clean", "smooth", "polished"
If all three are near normal, say NOTHING about texture. Silence is better than a filler line.
NEVER use these words: "dissonance", "spectral", "compression", "complexity".

BANNED PHRASES — these have been overused across hundreds of emails. If you write any of these, delete it and try again:
- "everything locked in tight"
- "everything locked in"
- "no rough edges anywhere"
- "no rough edges"
- "floaty and weightless up top but there's an edge"
- "lifts you up but keeps you unsettled"
- "hits hard but lifts you up"
- "sinks into you"
- "without being dramatic about it"
- "without trying too hard"
- "lot of low end" as your first three words
- "everything's polished"
- "lifts you up" — try "opens up", "takes you somewhere", "pulls you up"
- "edge running through it" — try "something unsettled", "tension underneath", "doesn't fully resolve"
- "keeps you on edge"
- "pulls you somewhere"
- "hits different"

TONE: You're texting a producer friend about a track you just heard. Short, casual, zero polish.
NO em dashes (—). Use commas and periods only. Zero em dashes.
Never say "space" (as in "the R&B space"). Never say "keeps the energy up" or "maintains".
Drop filler words. Contractions over formal. Fragments are fine.
Think iMessage, not LinkedIn.

CRITICAL RULES:
1. NEVER reference the artist, track title, or any proper nouns. Just describe the SOUND.
   BAD: "Jack's got a lot of low end" (uses artist name)
   BAD: "Peterson's remix has..." (uses artist name)
   BAD: "Zonard's got that heavy bottom" (uses track title)
   BAD: "Starting Block hits hard" (uses track title to start)
   BAD: "Got this French thing called Aria" (mentions track title)
   BAD: "This one called [title]" (references title)
   BAD: "The remix has..." (references track type from title)
   GOOD: "Bright and airy up top with just enough weight underneath"
   GOOD: "Super wide stereo image — everything's got its own pocket"
   GOOD: "Dark and moody, mids are doing all the heavy lifting"
   GOOD: "Stripped back and clean — lot of space in this mix"

   NEVER start with the track title. Start with sonic descriptors like "Got", "Heavy", "Lot of", "Hits", "Dark", "Bright", etc.

2. Always capitalize nationality adjectives: French, German, Italian, Spanish, etc.
   BAD: "french jazz", "german techno"
   GOOD: "French jazz", "German techno"
   If the track is in a non-English language (Arabic, Spanish, Korean, etc.), do NOT reference or quote any lyrics. Stick purely to sonic/production observations. You can mention the genre/origin ("Arabic pop", "Latin trap") but never pretend to understand lyrics you'd have to translate.

3. NEVER use these phrases — they sound corporate/formal:
   - "in the [genre] space" → just say "for [genre]"
   - "keeps the energy up" → "keeps it moving" or "doesn't let up"
   - "maintains" → cut it
   - "creates a sense of" → just name the feeling
   - "elements of" → cut it
   - "showcases" → cut it
   - "delivers" → cut it

4. NO HYPERBOLE. You're understating, not overselling.
   - NEVER use "really" — cut it every time
   - NEVER use "incredibly", "amazingly", "beautifully", "perfectly"
   - NEVER use "so much", "such a", "absolutely"
   - If it sounds like a compliment, rewrite it as a neutral observation
   BAD: "really lifts", "really opens up", "really hits"
   GOOD: "lifts", "opens up", "hits"

5. AVOID REPETITIVE PATTERNS — vary your vocabulary:
   - DON'T start every observation with "Bright and floaty" or "Bright and airy" — vary your openers
   - DON'T use "got this" or "got that" as a crutch — rephrase: "there's a", "something", or just state it directly
   - DON'T overuse "gentle" — try "soft", "easy", "light touch", "understated", "pulled back" instead
   - DON'T overuse "hits hard" — try "punches", "lands heavy", "got weight", "doesn't hold back"
   - NEVER say "way smoother/cleaner than most [genre]" — this is a crutch. If you want to compare to genre norms, be specific about WHAT is different, not just "smoother"
   - DON'T overuse "floaty" — try "weightless", "hovering", "suspended", "light"
   - DON'T default to "dense, got layers tucked everywhere" — try "lot packed in there", "busy mix", "stuff stacked on stuff", "thick arrangement"

GOOD (notice each one sounds DIFFERENT — varied structure, varied vocabulary):
- "Thick mids doing the heavy lifting — pulls you somewhere melancholy without trying too hard. Compressed but it still moves."
- "Stripped back and wide open — just voice and keys filling the space. Understated but it sticks with you."
- "Punchy low end, bright attack on the snare — whole thing just drives. Something restless underneath though."
- "Warm and round, nothing sharp anywhere — sinks into you. Lot of layers tucked in there if you listen close."
- "Forward and aggressive up top but the bottom stays controlled — tension all through it, never resolves."
- "Sparse production, lot of room — the vocal just sits there exposed. Sad without being dramatic about it."

6. DON'T repeat the same idea in different words. One description per concept, then move on.
   BAD: "Heavy bottom end with lot of low end" (same thing twice)
   BAD: "Bright and airy, light and floating" (same idea restated)
   GOOD: "Heavy bottom end, sits right under everything"

BAD:
- "Sub-heavy foundation with ethereal transcendence" (parroting GEMS terms)
- "Spectral complexity is dense, lot of layers going on" (uses technical field name)
- "Creates powerful tension" (analysis vocab, not mixer talk)
- "Muddy" or "muddy in a way that works" (muddy is always negative in mixing, never use it)
- "Heavy bottom end with lot of low end" (says the same thing twice — pick one)
- "Heavy bottom end that maintains warmth in the R&B space" (corporate, uses "maintains" and "space")
- "Keeps the energy up while delivering emotional depth" (LinkedIn post, not a text)
- "Bright and floaty but got that punch — harmonics are clean as hell, way smoother than most pop" (repetitive pattern — this is what every email sounds like, stop)
- "Got that gentle, wistful thing going" (overuses "got that" + "gentle" + "wistful")

OUTPUT FORMAT (JSON only):
{{"observation": "string"}} or {{"observation": null}}"""


MANAGER_MIXER_PROMPT = """Generate 1-2 sentences about this track that sound like a producer who builds conversion data into their mixes. Casual, specific, forward-looking.

YOU MUST COVER:
1. SONIC CHARACTER — what does the mix sound like? Lead with whatever is MOST distinctive. Same approach as a mixer observation.
2. CONVERSION ANGLE — frame one sonic choice as something that either helps or could help the track land harder with new listeners. This is about making the NEXT release convert better, not fixing the current one. Forward-looking, not corrective. Never say "fix", "problem", "issue", "weak", or "failing".

FRAMING: You're a producer who profiles tracks to see what production choices convert listeners to followers. You listened to this track and noticed something specific about the sound AND how it relates to conversion in this genre. You want to bake smarter production into the next one.

GOOD FRAMING:
- "what's driving the follow" not "what's wrong"
- "land harder" not "fix the conversion"
- "bake into the next one" not "correct this"
- "open up room for" not "the problem is"
- "strongest conversion lever in [genre] is X, and this track [has it / could lean into it more]"

BIO BLURB (already sent): {bio_blurb}
If the bio blurb mentions genre details, DO NOT repeat genre in your observation.
If the bio blurb is about an award, label, collab, origin, or location — you SHOULD reference genre in your observation BUT never repeat any country, city, or location name that already appears in the bio blurb.
If there is no bio blurb (null), include genre context.
If genre is "N/A" or unknown, do NOT guess or invent a genre.

GENRE NOTE: Use the artist's own genre tag as the primary genre. Reference artist genres are supplementary context only.

INPUT:
- track: {track}
- genre: {genre}
- reference_artist_genres: {reference_genres}
- sonic_signature_text: {sonic_signature_text}
- tonal_balance_description: {tonal_balance_description}
- brightness_character: {brightness_character}
- bass_character: {bass_character}
- emotional_signature: {emotional_signature}
- emotion_1: {emotion_1}
- emotion_2: {emotion_2}
- genre_deviation_score: {genre_deviation_score}
- how_squeezed (0-1, normal is 0.80): {compression_amount}
- how_busy (0-1, normal is 0.07): {spectral_complexity}
- how_rough (0-1, normal is 0.15): {dissonance}

DEVIATION SCORE GUIDE:
- Above 0.6 = doing something different for the genre — worth a nod
- Below 0.6 = don't mention

CRITICAL — TRANSLATE, DON'T PARROT:
The emotion/tonal fields use analysis vocabulary. Convert to how a producer actually talks:
- "ethereal" → "floaty", "airy", "spacey"
- "fat" → "thick", "full", "chunky"
- "warm" → "warm", "round", "smooth"
- "balanced" → "sits right", "everything's in its lane"
- "transcendence" → "lifts", "opens up"
- "powerful" → "hits hard", "punchy", "got weight"
- "tenderness" → "soft", "gentle", "delicate"
- "tension" → "edge", "grit", "something unsettled"
DON'T default to talking about bass. Lead with whatever the tonal_balance_description actually says.

TONE: You're texting a manager about their artist's track. Casual, specific, zero polish. You sound like a producer who happens to have data, not a data person who happens to know music.
NO em dashes (—). Use commas and periods only.
Never say "space" (as in "the R&B space"). Drop filler words. Contractions over formal. Fragments are fine.
Think iMessage, not pitch deck.

CRITICAL RULES:
1. NEVER reference the artist, track title, or any proper nouns. Just describe the SOUND + conversion angle.
2. Always capitalize nationality adjectives: French, German, Italian, etc.
3. NO HYPERBOLE. Understated, not overselling.
4. NEVER use "fix", "problem", "issue", "weak", "failing", "broken", "wrong". This is about making the next one land harder, not diagnosing problems.
5. MAX 2 sentences. Keep it tight.

GOOD:
- "Bright and forward up top, vocal cuts right through, that's the profile that tends to trigger the follow in pop. Bake a little more width into the low mids on the next one and it'd land even harder."
- "Heavy bottom, compressed, whole thing drives. In hip-hop that tonal weight converts well, the arrangement could open up a bit more to let the hook breathe and that's where you'd see more follows."
- "Warm and round, nothing sharp, just sinks in. The conversion lever for R&B is usually vocal clarity and this one's got it, just needs the arrangement to not compete with it."

BAD:
- "The track has a conversion problem in the low end" (corrective, uses "problem")
- "This needs fixing in the mix" (corrective)
- "The listener-to-follower ratio suggests weak production" (data-bro language, uses "weak")

OUTPUT FORMAT (JSON only):
{{"observation": "string"}} or {{"observation": null}}"""


LYRIC_OBSERVATION_PROMPT = """Generate ONE sentence about what these lyrics are doing — the feeling, narrative arc, or something unexpected you noticed.

LYRICS (snippet):
{lyrics}

LYRIC EMOTIONS: {lyric_emotion_1}, {lyric_emotion_2}, {lyric_emotion_3}, {lyric_emotion_4}

VOICE: You're a mixer/producer who noticed the lyrics while working on the track. Casual, observational, feeling-first. Not literary criticism, not academic.

APPROACH — blend these:
1. FEELING FIRST: Lead with the emotional read — what do the lyrics feel like?
2. STORY ARC: What's happening in the lyrics? Keep it brief.
3. CONTRAST: If something unexpected stands out (title vs content, tone vs words), mention it.

CRITICAL RULES:
1. ONE sentence, under 25 words
2. NEVER reference the artist or track by name
3. ALWAYS start with "Lyrics feel" — "Lyrics feel resigned but angry", "Lyrics feel dreamy and already fading", "Lyrics feel manic and chaotic"
4. INCLUDE a short lyric fragment in quotes with "..." around it — pick the most iconic or striking phrase
   - Keep fragments to 4-8 words MAX
   - NEVER cut a word in half — every word in the fragment must be complete
   - BAD: '...I got two left feet and no rh...' (cut mid-word)
   - GOOD: '...I got two left feet...' (clean cut between words)
   - NON-ENGLISH LYRICS: If the lyrics are in Arabic, Spanish, French, Portuguese, Korean, Japanese, or ANY non-English language:
     - TRANSLATE the lyric fragment to English. Quote the English translation, not the original.
     - After the translated fragment, add "(had to Google Translate that one)" in parentheses.
     - Example: Arabic lyrics "انت غلطة يا حبيبي" → '...you were a mistake, my love...' (had to Google Translate that one)
     - Example: Spanish lyrics "no puedo más con este dolor" → '...I can't take this pain anymore...' (had to Google Translate that one)
     - The "Lyrics feel..." sentence should still describe the emotional feel in English as normal.
     - If you can't confidently translate, return {{"lyric_observation": null}} instead of guessing.
5. Sound like you're describing what you heard to another producer, not writing a review
6. MAX ONE em dash (—) per sentence. Use commas or periods instead of chaining with dashes.

GOOD:
- "Lyrics feel resigned but angry — '...let my bones bubble with me...' choosing to stay and burn."
- "Lyrics feel dreamy and already fading — '...ya lost the number when you said it...' slipping away mid-memory."
- "Lyrics feel like he says he's over it but '...I look back at everything you did...' — definitely not."
- "Lyrics feel politically angry but heartbroken about it — '...start 'em young, then they've won...'"
- "Lyrics feel manic and chaotic — '...hot dogs and grenades...' everything spinning and totally fine with it."
- "Lyrics feel tired and sad — '...it's about to break down...' watching something crumble in slow motion."
- "Lyrics feel exhausted but finally peaceful, '...you were a mistake, my love...' (had to Google Translate that one) admitting everyone was right all along."

BAD:
- "The lyrics explore themes of loss and redemption" (academic, no lyric fragment)
- "Beautiful storytelling about heartbreak" (review/compliment)
- "This song is about a breakup" (boring, no feeling, no fragment)
- "'I set the house on fire I wouldn't leave I'd sit and let my bones bubble'" (too long a quote, needs "..." and trimming)

OUTPUT FORMAT (JSON only):
{{"lyric_observation": "string"}} or {{"lyric_observation": null}}"""


BIO_BLURB_PROMPT = """Extract ONE specific fact from this artist bio to use as a cold email opener line.

BIO: {bio}
LOCATION: {location}
GENRES: {genres}
ARTIST_NAME: {artist_name}
ADDRESSING: {addressing}

ADDRESSING MODE:
- If ADDRESSING is "artist": Use "you/you're/you've" — we're emailing the artist directly.
- If ADDRESSING is "manager": Use the artist's name instead of "you" — we're emailing their manager/rep.
  Examples for manager mode:
  - "Saw {artist_name} won [X] — congrats." (not "you won")
  - "Noticed {artist_name} is signed with [X]." (not "you're signed")
  - "Saw {artist_name} worked with [X]." (not "you worked")
  - "Saw {artist_name} is doing [genre] out of [place]." (not "you're doing")

GENRE RULE: ALWAYS weave the genre into the blurb when the GENRES field has something specific (not just "pop" or "rock").
- If you have a bio fact AND a genre, combine them: "Saw you worked with Ice Cube — hip-hop with that West Coast weight."
- If genre is unusual or a blend, lead with it: "You're mixing [genre] with [genre] — interesting combo."
- If genre name is bizarre/funny (barbecue pop, pirate metal, witch house), make a short witty remark.
- You're LISTENING to their music, not reading about it. Never say "Read" for genre observations.
- Only skip genre if it's too generic to be interesting (just "pop", "rock", "hip-hop" alone with nothing to add).

PRIORITY ORDER for the bio fact (combine with genre when possible):
1. Award/milestone → "Read you won [X] — congrats." / "Read {artist_name} won [X] — congrats."
2. Label signing → "Noticed you're signed with [X]." / "Noticed {artist_name} is signed with [X]."
3. Notable collab/feature → "Saw you worked with [X]." / "Saw {artist_name} worked with [X]."
4. Origin story (if interesting) → "Saw you came up in [place]." / "Saw {artist_name} came up in [place]."
5. Unique background detail (internet culture, specific scene, unusual path) → reference it naturally
6. Location + genre (non-US/AU/CA) → "Saw you're doing [genre] out of [place]." / "Saw {artist_name} is doing [genre] out of [place]."
   ⚠️ UK/GB OVERRIDE: If the artist is from the UK, ALWAYS use this format instead:
   "Saw you're out of [UK city] — I grew up in Knaresborough, North Yorkshire." / "Saw {artist_name} is out of [UK city] — I grew up in Knaresborough, North Yorkshire."
   NEVER use the generic location+genre format for UK artists. ALWAYS include the Knaresborough tie.
7. Longevity → LAST RESORT, only if specific year mentioned. "Read you've been at it since [year]." / "Read {artist_name} has been at it since [year]."
   NEVER use vague longevity like "more than a decade" or "for years" — skip to null instead.
8. Genre only (if no bio fact but genre is interesting) → "You're doing [genre] — interesting." / "{artist_name} is doing [genre] — interesting."

RULES:
- ONE or two short sentences, under 18 words total (artist name counts as one word)
- Use "Saw" or "Read" or "Noticed" to start — never "I saw" or "I read"
- No compliments, no adjectives like "amazing" or "great"
- No questions — this is a statement
- NEVER call a genre "bold" or "brave" or "unexpected" if it's common in that region. Death metal from Germany is not bold. K-pop from Korea is not unexpected. Reggaeton from Puerto Rico is not surprising. Only comment on genre being unusual if it genuinely is.
- NEVER reference the artist's own name, label, or project as if it's a separate entity. If the artist is "Doomtree" and the bio mentions "Doomtree Records" or "self-titled debut on Doomtree," that's THEIR OWN label/name — don't use it as a fact. Find something else.
- NEVER reference upcoming releases, tours, dates, or events — bio data may be stale. "Releasing new EP in 2024" will be wrong by the time the email sends. Stick to facts that don't expire (awards, collabs, origin, label, genre). Past accomplishments with years are OK ("won X in 2024"), but future/ongoing plans are NOT.
- NEVER mention holidays, religious themes, or seasonal genres (Christmas, Easter, Hanukkah, Ramadan, etc.). If the genre says "Christmas music" or similar, ignore it and use a different fact or skip to location+genre fallback.
- If the bio is empty, generic, just social links, or only has vague/philosophical language without concrete facts → skip to location+genre fallback
- Skip if bio is in a foreign language you can't parse for facts
- For location+genre fallback: SKIP if location is US, AU, CA
- UK/GB artists MUST ALWAYS get the Knaresborough format. This is mandatory, never skip the personal tie for UK artists.
- If ADDRESSING is "manager", NEVER use "you/you're/you've" — always use the artist's name.

GOOD (artist mode):
- "Read you won Mad Cool Talent 2024 — congrats."
- "Noticed you're signed with Def Jam."

GOOD (manager mode):
- "Read Soaked Oats won Mad Cool Talent 2024 — congrats."
- "Noticed Soaked Oats is signed with Def Jam."

BAD:
- "Your music is amazing" (compliment)
- "I saw you're from Atlanta" (starts with "I")
- "You've won awards and worked with many artists" (vague, multiple facts)
- Using "you" when ADDRESSING is "manager"

OUTPUT FORMAT (JSON only, no explanation):
{{"blurb": "string"}} or {{"blurb": null}}"""




def generate_bio_blurb(artist_data, recipient_info=None):
    """Generate a short bio blurb from artist description."""
    bio = artist_data.get('bio') or ''
    location_code = artist_data.get('code2')
    location = get_country_name(location_code)
    genres = artist_data.get('genres', 'N/A')
    artist_name = artist_data.get('name') or ''

    if recipient_info is None:
        recipient_info = {"type": "artist"}
    addressing = recipient_info.get("type", "artist")
    is_manager = addressing == "manager"

    # If no bio, try location+genre fallback for non-US/AU/CA
    if not bio or len(bio.strip()) < 20:
        skip_locations = ['US', 'AU', 'CA']
        if location_code and location_code.upper() not in skip_locations:
            # Strip demonym from genre if it matches country
            genre_clean = genres.split(',')[0].strip() if genres else ''
            # Remove nationality prefix if present
            for dem_country, demonyms in DEMONYMS.items():
                if location == dem_country:
                    for dem in demonyms:
                        if genre_clean.lower().startswith(dem.lower()):
                            genre_clean = genre_clean[len(dem):].strip()
                            break
            if genre_clean:
                if location_code and location_code.upper() in ['GB', 'UK']:
                    uk_loc = location if location != 'United Kingdom' else 'the UK'
                    if is_manager:
                        return f"Saw {artist_name} is out of {uk_loc} — I grew up in Knaresborough, North Yorkshire."
                    return f"Saw you're out of {uk_loc} — I grew up in Knaresborough, North Yorkshire."
                if is_manager:
                    return f"Saw {artist_name} is doing the {genre_clean} thing out of {location}."
                return f"Saw you're doing the {genre_clean} thing out of {location}."
        return None

    prompt = BIO_BLURB_PROMPT.format(
        bio=bio,
        location=location,
        genres=genres,
        artist_name=artist_name,
        addressing=addressing
    )

    response = claude.messages.create(
        model=MODEL_STRUCTURED,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text.strip()

    # Clean up markdown if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        result = json.loads(response_text)
        blurb = result.get('blurb')
        if not blurb:
            return None
        blurb = strip_demonyms(blurb)
        # Strip em dashes — LLM ignores the ban
        blurb = blurb.replace(' — ', ', ')
        # UK artists: strip redundant UK/British genre prefixes, append Knaresborough tie
        if location_code and location_code.upper() in ['GB', 'UK']:
            import re
            blurb = re.sub(r'\bthe UK\b', 'the UK', blurb)  # normalize first
            blurb = re.sub(r'\bthe United Kingdom\b', 'the UK', blurb)
            blurb = re.sub(r'\bUnited Kingdom\b', 'the UK', blurb)
            blurb = re.sub(r'out of the doing ', 'out of the UK doing ', blurb)  # fix orphaned "the"
            blurb = re.sub(r'\bUK\s+(?=hip-hop|rap|pop|rock|R&B|soul|grime|drill|dnb|garage|dubstep|dance|electronic|indie|folk|metal|punk|jazz|blues|country|reggae|dancehall|afrobeat)', '', blurb, flags=re.IGNORECASE)
            blurb = re.sub(r'\bBritish\s+(?=hip-hop|rap|pop|rock|R&B|soul|grime|drill|dnb|garage|dubstep|dance|electronic|indie|folk|metal|punk|jazz|blues|country|reggae|dancehall|afrobeat)', '', blurb, flags=re.IGNORECASE)
            if 'Knaresborough' not in blurb:
                blurb = blurb.rstrip('.') + '. I grew up in Knaresborough, North Yorkshire.'
        return blurb
    except json.JSONDecodeError:
        return None


def generate_mixer_observation(artist_data, bio_blurb=None, reference_genres=None, recipient_type='artist'):
    """Generate a mixer's sonic + emotional observation. Uses conversion-focused prompt for managers."""
    # Need sonic data to generate this
    if not artist_data.get('sonic_signature_text'):
        return None

    # Use conversion-focused prompt for manager recipients
    base_prompt = MANAGER_MIXER_PROMPT if recipient_type == 'manager' else MIXER_OBSERVATION_PROMPT

    prompt = base_prompt.format(
        bio_blurb=bio_blurb or 'null',
        track=artist_data.get('top_track', 'N/A'),
        genre=artist_data.get('genres', 'N/A'),
        reference_genres=reference_genres or 'N/A',
        sonic_signature_text=artist_data.get('sonic_signature_text', 'N/A'),
        tonal_balance_description=artist_data.get('tonal_balance_description', 'N/A'),
        brightness_character=artist_data.get('brightness_character', 'N/A'),
        bass_character=artist_data.get('bass_character', 'N/A'),
        emotional_signature=artist_data.get('emotional_signature', 'N/A'),
        emotion_1=artist_data.get('emotion_1', 'N/A'),
        emotion_2=artist_data.get('emotion_2', 'N/A'),
        genre_deviation_score=artist_data.get('genre_deviation_score', 'N/A'),
        compression_amount=artist_data.get('compression_amount', 'N/A'),
        spectral_complexity=artist_data.get('spectral_complexity', 'N/A'),
        dissonance=artist_data.get('dissonance', 'N/A')
    )

    response = claude.messages.create(
        model=MODEL_CREATIVE,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text.strip()

    # Clean up markdown if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        result = json.loads(response_text)
        obs = result.get('observation')
        if obs:
            # LLM ignores em dash ban — strip them programmatically
            obs = obs.replace(' — ', ', ').replace(' - ', ', ')
        return obs
    except json.JSONDecodeError:
        return None


def get_emotion_category(emotion):
    """Map a specific emotion to its root category."""
    if not emotion:
        return None
    if emotion in EMOTION_TO_CATEGORY:
        return EMOTION_TO_CATEGORY[emotion]
    for key, category in EMOTION_TO_CATEGORY.items():
        if key.lower() == emotion.lower():
            return category
    return None


def get_dominant_category(emotions):
    """Count categories across emotions list and return the dominant one."""
    from collections import Counter
    categories = [get_emotion_category(e) for e in emotions if e]
    categories = [c for c in categories if c]
    if not categories:
        return None
    return Counter(categories).most_common(1)[0][0]


def analyze_lyric_emotions_inline(lyrics_fragment):
    """Analyze lyrics and return emotion dict. Used inline when not pre-computed."""
    if not lyrics_fragment or len(lyrics_fragment) < 20:
        return None

    prompt = LYRIC_EMOTION_PROMPT.format(
        taxonomy=EMOTION_TAXONOMY,
        lyrics=lyrics_fragment[:800]
    )

    try:
        response = claude.messages.create(
            model=MODEL_STRUCTURED,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        if '```' in text:
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except Exception:
        return None


def derive_combined_emotion(music_emotions, lyric_emotions):
    """Derive combined compound emotion from 9x9 matrix using dominant categories."""
    music_cats = [music_emotions.get(f'emotion_{i}', '') for i in range(1, 5)]
    lyric_cats = [lyric_emotions.get(f'lyric_emotion_{i}', '') for i in range(1, 5)]

    music_dominant = get_dominant_category(music_cats)
    lyric_dominant = get_dominant_category(lyric_cats)

    if not music_dominant or not lyric_dominant:
        return None

    return COMPOUND_MATRIX.get(music_dominant, {}).get(lyric_dominant)


def ensure_lyric_emotions(artist_data):
    """If lyrics exist but emotions aren't computed, do it inline and save to DB."""
    lyrics = artist_data.get('lyrics_fragment')
    lyric_emotion_1 = artist_data.get('lyric_emotion_1')

    # Already computed or no lyrics
    if lyric_emotion_1 or not lyrics:
        return artist_data

    # Analyze inline
    emotions = analyze_lyric_emotions_inline(lyrics)
    if not emotions:
        return artist_data

    # Derive combined emotion
    music_emotions = {f'emotion_{i}': artist_data.get(f'emotion_{i}', '') for i in range(1, 5)}
    combined = derive_combined_emotion(music_emotions, emotions)

    # Save to tracks_lyrics DB
    isrc = artist_data.get('_isrc')
    if isrc:
        update_data = {
            'lyric_emotion_1': emotions.get('lyric_emotion_1'),
            'lyric_emotion_2': emotions.get('lyric_emotion_2'),
            'lyric_emotion_3': emotions.get('lyric_emotion_3'),
            'lyric_emotion_4': emotions.get('lyric_emotion_4'),
        }
        if combined:
            update_data['combined_emotion'] = combined
        supabase.table('tracks_lyrics').update(update_data).eq('isrc', isrc).execute()

    # Merge into artist_data
    artist_data.update(emotions)
    if combined:
        artist_data['combined_emotion'] = combined

    return artist_data


def generate_lyric_observation(artist_data):
    """Generate a feeling-first observation about the lyrics."""
    lyrics = artist_data.get('lyrics_fragment')
    lyric_emotion_1 = artist_data.get('lyric_emotion_1')

    if not lyrics or not lyric_emotion_1:
        return None

    prompt = LYRIC_OBSERVATION_PROMPT.format(
        lyrics=lyrics[:500],
        lyric_emotion_1=artist_data.get('lyric_emotion_1', 'N/A'),
        lyric_emotion_2=artist_data.get('lyric_emotion_2', 'N/A'),
        lyric_emotion_3=artist_data.get('lyric_emotion_3', 'N/A'),
        lyric_emotion_4=artist_data.get('lyric_emotion_4', 'N/A')
    )

    response = claude.messages.create(
        model=MODEL_CREATIVE,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text.strip()

    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        result = json.loads(response_text)
        obs = result.get('lyric_observation')
        if obs:
            obs = obs.replace(' — ', ', ').replace(' - ', ', ')
        return obs
    except json.JSONDecodeError:
        return None


def generate_combined_line(artist_data):
    """Generate the combined emotion line from music + lyrics data."""
    combined = artist_data.get('combined_emotion')
    if not combined:
        return None

    import random

    music_emotions = [artist_data.get(f'emotion_{i}', '') for i in range(1, 5)]
    lyric_emotions = [artist_data.get(f'lyric_emotion_{i}', '') for i in range(1, 5)]

    music_cat = get_dominant_category(music_emotions)
    lyric_cat = get_dominant_category(lyric_emotions)

    c = combined
    return c[0].upper() + c[1:] + ('.' if not c.endswith('.') else '')



def get_artist_data(artist_id, sp_artist_id=None):
    """Fetch artist data with GEMS analysis. sp_artist_id used to verify tracks belong to this artist."""
    # Get artist info
    artist_result = supabase.table('artists')\
        .select('id, name, description, code2, genres, moods, activities')\
        .eq('id', artist_id)\
        .execute()

    if not artist_result.data:
        return None

    artist = artist_result.data[0]

    # Get track data (both top and recent)
    tracks_result = supabase.table('tracks')\
        .select('top_track, isrc, recent_track, recent_track_isrc')\
        .eq('artist_id', artist_id)\
        .limit(1)\
        .execute()

    if not tracks_result.data:
        return {
            'name': artist.get('name'),
            'bio': artist.get('description'),
            'code2': artist.get('code2'),
            'genres': artist.get('genres'),
            'moods': artist.get('moods'),
            'activities': artist.get('activities'),
            'top_track': None,
            '_isrc': None,
        }

    track_row = tracks_result.data[0]
    top_name = track_row.get('top_track')
    top_isrc = track_row.get('isrc')
    recent_name = track_row.get('recent_track')
    recent_isrc = track_row.get('recent_track_isrc')

    # Verify tracks belong to this artist via Spotify ISRC lookup
    if sp_artist_id:
        top_owned = _verify_track_owner(top_isrc, sp_artist_id) if top_isrc else True
        rec_owned = _verify_track_owner(recent_isrc, sp_artist_id) if recent_isrc else True
        if not top_owned:
            print(f"⚠️ Top track '{top_name}' belongs to different Spotify artist — skipping it")
            top_name = None
            top_isrc = None
        if not rec_owned:
            print(f"⚠️ Recent track '{recent_name}' belongs to different Spotify artist — skipping it")
            recent_name = None
            recent_isrc = None

    # Check GEMS + lyrics availability for each track
    def check_track_data(isrc):
        if not isrc:
            return {}, {}, False, False
        gems = {}
        gems_result = supabase.table('gems_complete_analysis')\
            .select('sonic_signature_text, emotional_signature, bass_character, brightness_character, is_instrumental, danceability_description, tonal_balance_description, emotion_1, emotion_2, genre_deviation_score, compression_amount, spectral_complexity, dissonance')\
            .eq('isrc', isrc)\
            .execute()
        if gems_result.data:
            gems = gems_result.data[0]

        lyrics = {}
        lyrics_result = supabase.table('tracks_lyrics')\
            .select('lyrics_fragment, lyric_emotion_1, lyric_emotion_2, lyric_emotion_3, lyric_emotion_4, combined_emotion')\
            .eq('isrc', isrc)\
            .eq('has_lyrics', True)\
            .limit(1)\
            .execute()
        if lyrics_result.data:
            lyrics = lyrics_result.data[0]

        return gems, lyrics, bool(gems), bool(lyrics)

    top_gems, top_lyrics, top_has_gems, top_has_lyrics = check_track_data(top_isrc)
    rec_gems, rec_lyrics, rec_has_gems, rec_has_lyrics = check_track_data(recent_isrc)

    # Track selection — pick whichever track has the most complete data.
    # Score: GEMS = 1 point, lyrics = 1 point. Higher wins. Tie → recent track.
    # No GEMS = not usable (fall back to the other track).
    top_score = (1 if top_has_gems else 0) + (1 if top_has_lyrics else 0)
    rec_score = (1 if rec_has_gems else 0) + (1 if rec_has_lyrics else 0)

    # Recent wins if it has GEMS and at least as much data as top
    if rec_has_gems and rec_score >= top_score:
        selected_track = recent_name
        selected_isrc = recent_isrc
        gems_data = rec_gems
        lyrics_data = rec_lyrics if rec_has_lyrics else {}
    elif top_has_gems:
        selected_track = top_name
        selected_isrc = top_isrc
        gems_data = top_gems
        lyrics_data = top_lyrics if top_has_lyrics else {}
    elif rec_has_gems:
        # Top has no GEMS but recent does
        selected_track = recent_name
        selected_isrc = recent_isrc
        gems_data = rec_gems
        lyrics_data = rec_lyrics if rec_has_lyrics else {}
    else:
        # Neither track has GEMS — skip this artist
        return None

    return {
        'name': artist.get('name'),
        'bio': artist.get('description'),
        'code2': artist.get('code2'),
        'genres': artist.get('genres'),
        'moods': artist.get('moods'),
        'activities': artist.get('activities'),
        'top_track': selected_track,
        '_isrc': selected_isrc,
        '_track_type': 'recent' if selected_track == recent_name and recent_name else 'top',
        **gems_data,
        **lyrics_data
    }


def get_reference_genres(ref_name_1, ref_name_2):
    """Look up genres for reference artists by name."""
    ref_names = [n for n in [ref_name_1, ref_name_2] if n]
    if not ref_names:
        return None
    genres = set()
    for name in ref_names:
        result = supabase.table('artists')\
            .select('genres')\
            .ilike('name', name)\
            .limit(1)\
            .execute()
        if result.data and result.data[0].get('genres'):
            for g in result.data[0]['genres'].split(','):
                g = g.strip()
                if g:
                    genres.add(g)
    return ', '.join(sorted(genres)) if genres else None


def main(limit=None, min_listeners=None, skip_existing=False, artist_ids=None):
    print(f"Fetching prospects (limit: {limit or 'all'}, min listeners: {min_listeners or 'none'}, skip_existing: {skip_existing}, artist_ids: {artist_ids or 'all'})...")

    # 1. Build set of artist_ids already contacted (any row, any status)
    contacted_artist_ids = set()
    contacted_emails = set()
    last_id = 0
    batch_size = 1000

    while True:
        r = supabase.table('regression_prospects')\
            .select('id, artist_id, email')\
            .not_.is_('sequence_status', 'null')\
            .gt('id', last_id).order('id').limit(batch_size).execute()
        if not r.data:
            break
        for p in r.data:
            contacted_artist_ids.add(p['artist_id'])
            if p.get('email'):
                contacted_emails.add(p['email'].lower().strip())
        last_id = r.data[-1]['id']
        if len(r.data) < batch_size:
            break

    print(f"Found {len(contacted_artist_ids)} already-contacted artist_ids, {len(contacted_emails)} emails from regression_prospects")

    # 1b. ALSO check email_events for any prior sends (catches sends from deleted/recreated prospect rows)
    print("Loading email_events delivery history...")
    events_offset = 0
    events_page = 1000
    while True:
        r = supabase.table('email_events')\
            .select('email')\
            .eq('event_type', 'delivered')\
            .ilike('sequence_step', 'artist_outreach%')\
            .range(events_offset, events_offset + events_page)\
            .execute()
        if not r.data:
            break
        for e in r.data:
            if e.get('email'):
                contacted_emails.add(e['email'].lower().strip())
        if len(r.data) < events_page:
            break
        events_offset += len(r.data)

    print(f"Total contacted emails (including email_events): {len(contacted_emails)}")

    # 2. Get sendable prospects
    all_prospects = []
    last_id = 0

    while True:
        query = supabase.table('regression_prospects')\
            .select('id, artist_id, name, email, reference_artist_name, reference_artist_2_name, spotify_url')\
            .is_('sequence_status', 'null')\
            .eq('sequence_paused', False)\
            .gt('id', last_id)\
            .order('id')\
            .limit(batch_size)
        if artist_ids:
            query = query.in_('artist_id', artist_ids)
        if min_listeners:
            query = query.gte('spotify_monthly_listeners', min_listeners)
        if skip_existing:
            query = query.is_('bio_blurb', 'null')

        result = query.execute()
        if not result.data:
            break
        all_prospects.extend(result.data)
        last_id = result.data[-1]['id']
        if len(result.data) < batch_size:
            break

    # 3. Deduplicate by artist_id AND filter out already-contacted
    seen_artist_ids = set()
    seen_emails = set()
    unique_prospects = []
    dup_artist = 0
    already_contacted = 0
    dup_email = 0

    for p in all_prospects:
        aid = p['artist_id']
        email = (p.get('email') or '').lower().strip()

        if aid in contacted_artist_ids:
            already_contacted += 1
            continue
        if aid in seen_artist_ids:
            dup_artist += 1
            continue
        if email and email in contacted_emails:
            dup_email += 1
            continue
        if email and email in seen_emails:
            dup_email += 1
            continue

        seen_artist_ids.add(aid)
        if email:
            seen_emails.add(email)
        unique_prospects.append(p)

    skipped = dup_artist + already_contacted + dup_email

    # Pre-filter: only keep prospects whose artist has GEMS data
    if unique_prospects:
        gems_artist_ids = set()
        prospect_aids = [p['artist_id'] for p in unique_prospects]
        for batch_start in range(0, len(prospect_aids), 200):
            batch = prospect_aids[batch_start:batch_start + 200]
            tr = supabase.table('tracks').select('artist_id, isrc').in_('artist_id', batch).execute()
            if tr.data:
                isrcs = [t['isrc'] for t in tr.data if t.get('isrc')]
                for isrc_batch_start in range(0, len(isrcs), 200):
                    isrc_batch = isrcs[isrc_batch_start:isrc_batch_start + 200]
                    gr = supabase.table('gems_complete_analysis').select('isrc').in_('isrc', isrc_batch).not_.is_('sonic_signature_text', 'null').execute()
                    if gr.data:
                        gems_isrcs = {g['isrc'] for g in gr.data}
                        for t in tr.data:
                            if t.get('isrc') in gems_isrcs:
                                gems_artist_ids.add(t['artist_id'])
        no_gems = len(unique_prospects) - len([p for p in unique_prospects if p['artist_id'] in gems_artist_ids])
        unique_prospects = [p for p in unique_prospects if p['artist_id'] in gems_artist_ids]
        print(f"GEMS pre-filter: {no_gems} skipped (no GEMS data), {len(unique_prospects)} remaining")

    # Apply limit after deduplication if specified
    if limit and len(unique_prospects) > limit:
        unique_prospects = unique_prospects[:limit]

    total = len(unique_prospects)
    print(f"Found {total} truly unique prospects ({skipped} skipped: {already_contacted} already contacted, {dup_artist} dup artist_id, {dup_email} dup email)\n")

    success_count = 0
    error_count = 0
    no_data_count = 0

    for i, prospect in enumerate(unique_prospects, 1):
        prospect_id = prospect['id']
        artist_id = prospect['artist_id']
        name = prospect['name']

        print(f"[{i}/{total}] Processing {name}...", end=" ", flush=True)

        # Get artist data — extract Spotify artist ID for track verification
        sp_url = prospect.get('spotify_url') or ''
        sp_artist_id = sp_url.split('/artist/')[-1].split('?')[0] if '/artist/' in sp_url else None
        artist_data = get_artist_data(artist_id, sp_artist_id=sp_artist_id)

        if not artist_data or not artist_data.get('top_track'):
            print("⚠️ No track data")
            no_data_count += 1
            continue

        try:
            # Detect if emailing artist directly or their manager/rep
            recipient_info = detect_email_recipient(prospect.get('email'), name)

            # Ensure lyric emotions are computed (inline if needed)
            artist_data = ensure_lyric_emotions(artist_data)

            # Generate bio blurb first (mixer needs to know what bio said)
            bio_blurb = generate_bio_blurb(artist_data, recipient_info=recipient_info)

            # Get reference artist genres for more accurate genre context
            ref_genres = get_reference_genres(
                prospect.get('reference_artist_name'),
                prospect.get('reference_artist_2_name')
            )

            # Generate mixer observation (pass bio_blurb to avoid genre redundancy)
            # For managers: uses conversion-focused prompt instead of pure sonic observation
            mixer_obs = generate_mixer_observation(artist_data, bio_blurb=bio_blurb, reference_genres=ref_genres, recipient_type=recipient_info.get('type', 'artist'))

            # Generate lyric observation (if lyrics available)
            lyric_obs = generate_lyric_observation(artist_data)

            # Generate combined emotion line
            combined_line = generate_combined_line(artist_data)

            if bio_blurb or mixer_obs:
                update_data = {}
                if bio_blurb:
                    update_data['bio_blurb'] = bio_blurb
                if mixer_obs:
                    update_data['mixer_observation'] = mixer_obs
                if lyric_obs:
                    update_data['lyric_observation'] = lyric_obs
                if combined_line:
                    update_data['combined_emotion'] = combined_line

                # Store recipient detection results
                update_data['recipient_type'] = recipient_info.get('type', 'artist')
                update_data['recipient_name'] = recipient_info.get('name')

                # Build greeting (first name for people, full name for bands)
                if recipient_info.get('type') == 'manager' and recipient_info.get('name'):
                    update_data['greeting'] = f"Hey {recipient_info['name']},"
                else:
                    update_data['greeting'] = f"Hey {shorten_artist_name(name)},"

                # Store selected track name + ISRC + type so send script uses the right one
                selected_track = artist_data.get('top_track')
                selected_isrc = artist_data.get('_isrc')
                track_type = artist_data.get('_track_type', 'top')
                if selected_track:
                    update_data['target_track_name'] = selected_track
                    update_data['target_track_type'] = f'{track_type}_track'
                if selected_isrc:
                    update_data['target_track_isrc'] = selected_isrc

                supabase.table('regression_prospects')\
                    .update(update_data)\
                    .eq('id', prospect_id)\
                    .execute()

                recipient_label = f" | to: {recipient_info.get('type')}" + (f" ({recipient_info.get('name')})" if recipient_info.get('name') and recipient_info.get('type') == 'manager' else "")
                track_label = f" | track: {track_type}" if selected_track else ""
                blurb_indicator = f" | blurb: ✓" if bio_blurb else ""
                mixer_indicator = f" | mixer: ✓" if mixer_obs else ""
                lyric_indicator = f" | lyric: ✓" if lyric_obs else ""
                combined_indicator = f" | combo: ✓" if combined_line else ""
                print(f"✓{blurb_indicator}{mixer_indicator}{lyric_indicator}{combined_indicator}{track_label}{recipient_label}")
                # Show generated content
                refs = [prospect.get('reference_artist_name'), prospect.get('reference_artist_2_name')]
                refs = [r for r in refs if r]
                ref_str = f"  refs: {', '.join(refs)}" if refs else ""
                print(f"  track: \"{selected_track}\"")
                if bio_blurb:
                    print(f"  blurb: {bio_blurb}")
                if mixer_obs:
                    print(f"  mixer: {mixer_obs}")
                if lyric_obs:
                    print(f"  lyric: {lyric_obs}")
                if combined_line:
                    print(f"  combo: {combined_line}")
                if ref_str:
                    print(ref_str)
                print()
                success_count += 1
            else:
                print("⚠️ No data generated")
                no_data_count += 1

            # Rate limit
            time.sleep(1.0)

        except Exception as e:
            print(f"✗ Error: {e}")
            error_count += 1
            time.sleep(1)

    print(f"\n{'='*60}")
    print(f"Complete!")
    print(f"  Success: {success_count}")
    print(f"  No data/hook: {no_data_count}")
    print(f"  Errors: {error_count}")


def regen_bio_and_mixer(limit=None):
    """Regenerate only bio_blurb and mixer_observation for unsent prospects that already have them."""
    print("=== REGEN MODE: bio_blurb + mixer_observation only ===\n")

    # Get unsent prospects that already have a mixer_observation
    all_prospects = []
    last_id = 0
    batch_size = 500

    while True:
        result = supabase.table('regression_prospects')\
            .select('id, artist_id, name, email, spotify_url')\
            .is_('sequence_status', 'null')\
            .eq('sequence_paused', False)\
            .not_.is_('mixer_observation', 'null')\
            .not_.is_('email', 'null')\
            .gt('id', last_id)\
            .order('id')\
            .limit(batch_size)\
            .execute()
        if not result.data:
            break
        all_prospects.extend(result.data)
        last_id = result.data[-1]['id']
        if len(result.data) < batch_size:
            break

    if limit and len(all_prospects) > limit:
        all_prospects = all_prospects[:limit]

    total = len(all_prospects)
    print(f"Found {total} unsent prospects with existing mixer_observation to regenerate\n")

    success_count = 0
    error_count = 0
    no_data_count = 0

    for i, prospect in enumerate(all_prospects, 1):
        prospect_id = prospect['id']
        artist_id = prospect['artist_id']
        name = prospect['name']

        print(f"[{i}/{total}] Regen {name}...", end=" ", flush=True)

        sp_url = prospect.get('spotify_url') or ''
        sp_artist_id = sp_url.split('/artist/')[-1].split('?')[0] if '/artist/' in sp_url else None
        artist_data = get_artist_data(artist_id, sp_artist_id=sp_artist_id)
        if not artist_data or not artist_data.get('top_track'):
            print("⚠️ No track data")
            no_data_count += 1
            continue

        try:
            recipient_info = detect_email_recipient(prospect.get('email'), name)

            bio_blurb = generate_bio_blurb(artist_data, recipient_info=recipient_info)
            mixer_obs = generate_mixer_observation(artist_data, bio_blurb=bio_blurb, recipient_type=recipient_info.get('type', 'artist'))

            if mixer_obs:
                update_data = {'mixer_observation': mixer_obs}
                if bio_blurb:
                    update_data['bio_blurb'] = bio_blurb
                update_data['recipient_type'] = recipient_info.get('type', 'artist')
                update_data['recipient_name'] = recipient_info.get('name')
                if recipient_info.get('type') == 'manager' and recipient_info.get('name'):
                    update_data['greeting'] = f"Hey {recipient_info['name']},"
                else:
                    update_data['greeting'] = f"Hey {shorten_artist_name(name)},"

                supabase.table('regression_prospects')\
                    .update(update_data)\
                    .eq('id', prospect_id)\
                    .execute()

                recipient_label = f" | to: {recipient_info.get('type')}" + (f" ({recipient_info.get('name')})" if recipient_info.get('name') and recipient_info.get('type') == 'manager' else "")
                blurb_indicator = f" | blurb: ✓" if bio_blurb else ""
                print(f"✓{blurb_indicator} | mixer: ✓{recipient_label}")
                success_count += 1
            else:
                print("⚠️ No mixer generated")
                no_data_count += 1

            time.sleep(1.0)

        except Exception as e:
            print(f"✗ Error: {e}")
            error_count += 1
            time.sleep(1)

    print(f"\n{'='*60}")
    print(f"Regen complete!")
    print(f"  Success: {success_count}")
    print(f"  No data: {no_data_count}")
    print(f"  Errors: {error_count}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--regen-mixer':
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        regen_bio_and_mixer(limit=limit)
    else:
        limit = None
        min_listeners = None
        skip_existing = False
        artist_ids = None
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            if args[i] == '--min-listeners' and i + 1 < len(args):
                min_listeners = int(args[i + 1])
                i += 2
            elif args[i] == '--skip-existing':
                skip_existing = True
                i += 1
            elif args[i] == '--artist-ids' and i + 1 < len(args):
                artist_ids = [int(x) for x in args[i + 1].split(',')]
                i += 2
            elif not args[i].startswith('--'):
                limit = int(args[i])
                i += 1
            else:
                i += 1
        main(limit=limit, min_listeners=min_listeners, skip_existing=skip_existing, artist_ids=artist_ids)
