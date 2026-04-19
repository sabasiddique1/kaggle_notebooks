# Production 500 Error Analysis

## Current Issue
Getting 500 Internal Server Error on `/api/auth/login` endpoint in production (Vercel), but works fine locally.

## Root Cause Analysis

### What We've Implemented:
1. ✅ **Database Backend**: Migrated from JSON files to SQLite database
2. ✅ **User Storage**: Using `UserModel` table in database
3. ✅ **Database Path**: `/tmp/emergencycare.db` on Vercel
4. ✅ **Error Handling**: Added comprehensive error handling

### Possible Causes:

#### 1. **Database Initialization Issue** ⚠️
- Database might not be initialized before first query
- Tables might not exist when login is called
- **Solution**: Added `init_db()` check in `get_user_by_email()`

#### 2. **Database File Permissions** ⚠️
- `/tmp` directory might not be writable
- Database file might not persist between requests
- **Solution**: Using SQLite which handles file creation automatically

#### 3. **Import Errors** ⚠️
- Missing dependencies on Vercel
- Circular import issues
- **Solution**: Added try-except around imports

#### 4. **JWT Token Creation Failure** ⚠️
- `JWT_SECRET_KEY` not set properly
- Token encoding fails
- **Solution**: Added error handling around token creation

#### 5. **Session Management** ⚠️
- Database session not properly closed
- Connection pool exhaustion
- **Solution**: Using context managers (try-finally)

## Debug Steps

### Step 1: Check Debug Endpoint
After deployment, visit:
```
https://emergency-care-navigator.vercel.app/api/debug/auth
```

This will show:
- Vercel environment status
- Database connection status
- User count
- JWT secret key status
- Any database errors

### Step 2: Check Vercel Logs
1. Go to Vercel Dashboard
2. Your Project → Logs
3. Look for errors with full traceback
4. Check for:
   - Database connection errors
   - Import errors
   - Authentication errors

### Step 3: Test Login Endpoint Directly
```bash
curl -X POST https://emergency-care-navigator.vercel.app/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"patient@demo.com","password":"patient123"}' \
  -v
```

The response will now include:
- Error message
- Error type
- Full traceback (for debugging)

## What We've Fixed:

1. ✅ **Added Debug Endpoint** (`/api/debug/auth`)
   - Shows database status
   - Shows user count
   - Shows configuration

2. ✅ **Improved Error Handling**
   - All errors return JSON (not HTML)
   - Full traceback in logs
   - Error type included in response

3. ✅ **Database Initialization**
   - Check and initialize database in `get_user_by_email()`
   - Initialize on startup event
   - Initialize in login endpoint

4. ✅ **Better Logging**
   - Log every step of login process
   - Log database errors with traceback
   - Log authentication errors

## Next Steps:

1. **Deploy and Check Debug Endpoint**
   - Visit `/api/debug/auth` to see what's wrong

2. **Check Vercel Logs**
   - Look for the actual error message
   - Share the error details

3. **Test Login**
   - Try login again
   - Check the error response (now includes details)

4. **Common Fixes Based on Error**:
   - If "no such table: users" → Database not initialized
   - If "database is locked" → Connection issue
   - If "JWT_SECRET_KEY" error → Set environment variable
   - If import error → Missing dependency

## Expected Behavior:

**On First Request (Cold Start)**:
1. Database initialized
2. Tables created
3. Demo users created (if none exist)
4. Login succeeds

**On Subsequent Requests**:
1. Database already exists
2. Users already exist
3. Login succeeds immediately

---

**After deployment, check `/api/debug/auth` endpoint to see what's actually happening!**

