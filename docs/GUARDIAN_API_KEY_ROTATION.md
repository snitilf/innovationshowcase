# Guardian API Key Rotation Guide

## Overview

The Guardian API has a rate limit of **500 requests per day per API key**. To collect data for all 19 countries across 7 years (2010-2016), we need approximately **532 API calls** (19 countries × 7 years × 4 chunks/year).

With a single API key, this would take **3-4 days**. By using multiple API keys with automatic rotation, you can complete the collection much faster.

## How It Works

1. **Automatic Rotation**: The system automatically rotates between API keys when one reaches the 500 request limit
2. **Usage Tracking**: Each API key's usage is tracked per day in `data/sentiment/guardian_api_usage.json`
3. **Smart Selection**: The system always uses the API key with the most remaining requests for the day

## Getting Additional API Keys

1. Go to https://open-platform.theguardian.com/access/
2. Sign up for a free account (or use existing account)
3. Request an API key
4. Copy the API key (format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

**Note**: You can have multiple API keys from the same account or use different accounts.

## Usage

### Basic Usage (Default Keys)

The system comes with 2 default API keys. Just run:

```bash
python3 src/sentiment_analysis.py \
    --provider auto \
    --start-year 2010 \
    --end-year 2016 \
    --pause 1.1 \
    --gdelt-max-records 100
```

### Using Additional API Keys

To add more API keys, use the `--guardian-api-keys` argument:

```bash
python3 src/sentiment_analysis.py \
    --provider auto \
    --start-year 2010 \
    --end-year 2016 \
    --pause 1.1 \
    --gdelt-max-records 100 \
    --guardian-api-keys \
        key1-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
        key2-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
        key3-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### Custom Request Limit

If your API keys have different limits (e.g., paid tier with 5000 requests/day), adjust with:

```bash
--guardian-max-requests 5000
```

## Monitoring API Usage

Check API key usage anytime:

```bash
cat data/sentiment/guardian_api_usage.json
```

Example output:
```json
{
  "93c1ada7": {
    "api_key": "93c1ada7-8c9d-4b38-aa03-20b53f43a1cb",
    "requests": {
      "2025-11-16": 487
    },
    "total_requests": 487
  },
  "c1bb1584": {
    "api_key": "c1bb1584-e60d-4d7a-b019-276f96dd0a53",
    "requests": {
      "2025-11-16": 13
    },
    "total_requests": 13
  }
}
```

## Speed Calculation

With **N API keys**, you can make **N × 500 requests per day**.

- **1 key**: 500 requests/day → ~3-4 days for full collection
- **2 keys**: 1000 requests/day → ~1.5-2 days for full collection  
- **3 keys**: 1500 requests/day → ~1 day for full collection
- **4 keys**: 2000 requests/day → ~12 hours for full collection

**Note**: Still need to respect 1 request/second rate limit, so actual time depends on how many requests per country-year are needed.

## Troubleshooting

### All Keys Exhausted

If all API keys reach their daily limit, the system will:
1. Log an error message
2. Stop collection for that country
3. Resume the next day (usage resets at midnight UTC)

### Authentication Errors

If you see `401` or `403` errors:
- Check that the API key is correct
- Verify the API key hasn't been revoked
- The system will automatically skip invalid keys and try the next one

### Rate Limit Errors (429)

If you see `429` errors:
- The system automatically marks that key as exhausted
- Rotates to the next available key
- Continues collection seamlessly

## Best Practices

1. **Start with 2-3 keys**: This is usually sufficient for most collections
2. **Monitor usage**: Check `guardian_api_usage.json` periodically
3. **Plan ahead**: If you know you'll need many requests, get keys in advance
4. **Respect limits**: Don't try to bypass rate limits - the system handles this automatically

