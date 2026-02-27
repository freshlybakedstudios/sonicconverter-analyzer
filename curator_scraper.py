"""
Curator email scraping via RapidAPI chain:
  Instagram → Facebook → Website contacts

Same RapidAPI endpoints as the main pipeline scrapers.
Populates progressively — returns as soon as an email is found at any stage.
"""

import os
import re
import time
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# RapidAPI endpoints
INSTAGRAM_API_HOST = "instagram-scraper-20251.p.rapidapi.com"
INSTAGRAM_API_URL = f"https://{INSTAGRAM_API_HOST}/userinfo/"
INSTAGRAM_RATE = 0.8

FACEBOOK_API_HOST = "facebook-scraper3.p.rapidapi.com"
FACEBOOK_API_URL = f"https://{FACEBOOK_API_HOST}/page/details"
FACEBOOK_RATE = 0.2

WEBSITE_API_HOST = "website-contacts-scraper.p.rapidapi.com"
WEBSITE_API_URL = f"https://{WEBSITE_API_HOST}/scrape-contacts"
WEBSITE_RATE = 0.35

EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')


def _get_rapidapi_key() -> Optional[str]:
    return os.getenv('RAPIDAPI_KEY')


def _extract_emails(text: str) -> list[str]:
    """Extract email addresses from text, filtering common non-human ones."""
    if not text:
        return []
    found = EMAIL_REGEX.findall(text)
    # Filter out common junk
    junk = {'noreply', 'no-reply', 'support', 'info@example', 'test@', 'admin@'}
    return [
        e for e in found
        if not any(j in e.lower() for j in junk)
        and not e.endswith('.png')
        and not e.endswith('.jpg')
    ]


def _scrape_instagram(username: str) -> dict:
    """Scrape Instagram profile for email and linked website."""
    api_key = _get_rapidapi_key()
    if not api_key:
        return {}

    try:
        time.sleep(INSTAGRAM_RATE)
        resp = requests.get(
            INSTAGRAM_API_URL,
            params={"username": username},
            headers={
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": INSTAGRAM_API_HOST,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return {}

        data = resp.json()
        result = {}

        # Direct email from profile
        email = data.get('public_email') or data.get('email') or ''
        if email:
            result['email'] = email
            result['email_source'] = 'instagram'

        # Bio may contain email
        bio = data.get('biography') or ''
        bio_emails = _extract_emails(bio)
        if bio_emails and 'email' not in result:
            result['email'] = bio_emails[0]
            result['email_source'] = 'instagram_bio'

        # External URL for website scraping
        ext_url = data.get('external_url') or ''
        if ext_url:
            result['website'] = ext_url

        # Facebook link
        fb = data.get('facebook_url') or ''
        if fb:
            result['facebook_url'] = fb

        return result

    except Exception as e:
        logger.debug(f"Instagram scrape failed for {username}: {e}")
        return {}


def _scrape_facebook(page_url: str) -> dict:
    """Scrape Facebook page for email."""
    api_key = _get_rapidapi_key()
    if not api_key:
        return {}

    try:
        time.sleep(FACEBOOK_RATE)
        resp = requests.get(
            FACEBOOK_API_URL,
            params={"url": page_url},
            headers={
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": FACEBOOK_API_HOST,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return {}

        data = resp.json()
        result = {}

        email = data.get('email') or ''
        if email:
            result['email'] = email
            result['email_source'] = 'facebook'

        # About text may have email
        about = data.get('about') or data.get('description') or ''
        about_emails = _extract_emails(about)
        if about_emails and 'email' not in result:
            result['email'] = about_emails[0]
            result['email_source'] = 'facebook_about'

        website = data.get('website') or ''
        if website:
            result['website'] = website

        return result

    except Exception as e:
        logger.debug(f"Facebook scrape failed for {page_url}: {e}")
        return {}


def _scrape_website(url: str) -> dict:
    """Scrape a website for contact emails."""
    api_key = _get_rapidapi_key()
    if not api_key:
        return {}

    try:
        time.sleep(WEBSITE_RATE)
        resp = requests.get(
            WEBSITE_API_URL,
            params={"url": url},
            headers={
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": WEBSITE_API_HOST,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            return {}

        data = resp.json()
        result = {}

        # Emails from response
        emails = data.get('emails') or []
        if isinstance(emails, list) and emails:
            filtered = _extract_emails(' '.join(emails))
            if filtered:
                result['email'] = filtered[0]
                result['email_source'] = 'website'

        return result

    except Exception as e:
        logger.debug(f"Website scrape failed for {url}: {e}")
        return {}


def scrape_curator_emails(curator_name: str, instagram_url: str = '',
                          facebook_url: str = '', website_url: str = '') -> dict:
    """
    Run the Instagram → Facebook → Website scraping chain.
    Returns as soon as an email is found at any stage.

    Returns dict with: {email, email_source, instagram, facebook, website}
    """
    result = {'name': curator_name}

    # Step 1: Instagram
    if instagram_url:
        # Extract username from URL
        username = instagram_url.rstrip('/').split('/')[-1].split('?')[0]
        if username:
            ig_data = _scrape_instagram(username)
            if ig_data.get('email'):
                result['email'] = ig_data['email']
                result['email_source'] = ig_data.get('email_source', 'instagram')
                return result
            if ig_data.get('facebook_url') and not facebook_url:
                facebook_url = ig_data['facebook_url']
            if ig_data.get('website') and not website_url:
                website_url = ig_data['website']

    # Step 2: Facebook
    if facebook_url:
        fb_data = _scrape_facebook(facebook_url)
        if fb_data.get('email'):
            result['email'] = fb_data['email']
            result['email_source'] = fb_data.get('email_source', 'facebook')
            return result
        if fb_data.get('website') and not website_url:
            website_url = fb_data['website']

    # Step 3: Website
    if website_url:
        web_data = _scrape_website(website_url)
        if web_data.get('email'):
            result['email'] = web_data['email']
            result['email_source'] = web_data.get('email_source', 'website')
            return result

    # Try searching Instagram by curator name as last resort
    if not instagram_url and curator_name:
        ig_data = _scrape_instagram(curator_name.replace(' ', '').lower())
        if ig_data.get('email'):
            result['email'] = ig_data['email']
            result['email_source'] = ig_data.get('email_source', 'instagram')
            return result

    return result
