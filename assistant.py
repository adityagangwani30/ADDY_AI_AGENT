from google_auth_oauthlib.flow import InstalledAppFlow

from config import CLIENT_SECRET_FILE, GOOGLE_SCOPES

flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), GOOGLE_SCOPES)
creds = flow.run_local_server(port=8080)

print("Refresh Token:", creds.refresh_token)
