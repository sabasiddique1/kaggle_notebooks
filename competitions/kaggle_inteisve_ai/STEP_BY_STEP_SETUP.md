# 🚀 Step-by-Step Setup Guide

Follow these steps **in order** to get your login working in production.

## Step 1: Open Vercel Dashboard

1. Go to https://vercel.com
2. Sign in to your account
3. Find and click on your project: **emergency-care-navigator** (or whatever it's named)

## Step 2: Navigate to Environment Variables

1. Click on the **Settings** tab (top navigation)
2. In the left sidebar, click **Environment Variables**

## Step 3: Add DATABASE_URL

1. Click the **Add New** button
2. Fill in the form:
   - **Key**: `DATABASE_URL`
   - **Value**: `postgresql://postgres:NAgyxHiwctATJwro@db.lgbpmgaacqawvfavtdzu.supabase.co:5432/postgres`
   - **Environment**: Check all three boxes:
     - ✅ Production
     - ✅ Preview  
     - ✅ Development
3. Click **Save**

**You should now see `DATABASE_URL` in your environment variables list.**

## Step 4: Verify JWT_SECRET_KEY is Set

1. Scroll through your environment variables list
2. Look for `JWT_SECRET_KEY`
3. If it's **NOT there**, add it:
   - Click **Add New**
   - **Key**: `JWT_SECRET_KEY`
   - **Value**: Generate one by running this in terminal:
     ```bash
     python3 -c "import secrets; print(secrets.token_urlsafe(32))"
     ```
   - Copy the output and paste as the value
   - Select all environments
   - Click **Save**

## Step 5: Redeploy Your Project

**Option A: Automatic Redeploy (Recommended)**
- Just wait! Vercel will auto-deploy when you push code (which we already did)
- Go to **Deployments** tab and watch for a new deployment

**Option B: Manual Redeploy**
1. Go to **Deployments** tab
2. Find the latest deployment
3. Click the **⋯** (three dots) menu
4. Click **Redeploy**
5. Wait for deployment to complete (usually 1-2 minutes)

## Step 6: Verify Database Connection

1. Wait for deployment to finish (check the **Deployments** tab)
2. Once deployed, visit this URL (replace with your actual domain):
   ```
   https://emergency-care-navigator.vercel.app/api/debug/auth
   ```
   Or:
   ```
   https://emergency-care-navigator-n2ur.vercel.app/api/debug/auth
   ```

3. You should see JSON like this:
   ```json
   {
     "database_type": "PostgreSQL",
     "database_accessible": true,
     "user_count": 3,
     "vercel": true,
     "database_url_set": true
   }
   ```

**✅ If you see `"database_type": "PostgreSQL"` and `"database_accessible": true`, you're good!**

## Step 7: Test Login

1. Go to your app: `https://emergency-care-navigator.vercel.app`
2. Click **Login** or **Sign In**
3. Use these credentials:
   - **Email**: `patient@demo.com`
   - **Password**: `patient123`
4. Click **Login**

**✅ If login works, you're done!**

## 🆘 Troubleshooting

### If `/api/debug/auth` shows `"database_type": "SQLite"`
- **Problem**: `DATABASE_URL` is not set or not being read
- **Fix**: 
  1. Go back to **Settings** → **Environment Variables**
  2. Verify `DATABASE_URL` exists and is spelled correctly
  3. Make sure all environments are selected
  4. Redeploy

### If `/api/debug/auth` shows `"database_accessible": false`
- **Problem**: Database connection failed
- **Fix**:
  1. Check Vercel logs: **Deployments** → Click latest deployment → **Functions** tab
  2. Look for database connection errors
  3. Verify the connection string is correct (no extra spaces)
  4. Check Supabase dashboard to ensure database is running

### If login still returns 500 error
- **Problem**: Database not initialized or connection issue
- **Fix**:
  1. Check Vercel logs for detailed error messages
  2. Visit `/api/debug/auth` to see what's wrong
  3. Make sure `DATABASE_URL` is set correctly
  4. Try redeploying again

### If you see "Unexpected token" error
- **Problem**: Server returning HTML instead of JSON
- **Fix**: This should be fixed now, but if it happens, check Vercel logs

## ✅ Success Checklist

- [ ] `DATABASE_URL` added in Vercel environment variables
- [ ] `JWT_SECRET_KEY` set in Vercel environment variables
- [ ] Project redeployed
- [ ] `/api/debug/auth` shows `"database_type": "PostgreSQL"`
- [ ] `/api/debug/auth` shows `"database_accessible": true`
- [ ] Login works with `patient@demo.com` / `patient123`

---

**Once all checkboxes are checked, your login will work reliably!** 🎉

