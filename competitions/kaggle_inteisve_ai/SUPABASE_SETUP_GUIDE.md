# 🗄️ Supabase Database Setup Guide

Complete guide to set up Supabase PostgreSQL database for your project.

## Step 1: Create Supabase Account & Project

1. **Go to Supabase**
   - Visit: https://supabase.com
   - Click **Start your project** or **Sign in**

2. **Sign Up / Sign In**
   - Use GitHub, Google, or email
   - Free tier is sufficient for this project

3. **Create New Project**
   - Click **New Project** (or **+ New Project**)
   - Fill in:
     - **Name**: `emergency-care-navigator` (or any name)
     - **Database Password**: **IMPORTANT** - Save this password! You'll need it.
       - Example: `NAgyxHiwctATJwro` (you already have this)
     - **Region**: Choose closest to your users (e.g., `US East`, `EU West`)
     - **Pricing Plan**: Free (for development)
   - Click **Create new project**

4. **Wait for Project Setup**
   - Takes 1-2 minutes
   - You'll see "Setting up your project..." message
   - Wait until you see "Project is ready"

## Step 2: Get Database Connection String

1. **Go to Project Settings**
   - In your Supabase dashboard, click **Settings** (gear icon) in left sidebar
   - Click **Database** under Project Settings

2. **Find Connection String**
   - Scroll to **Connection string** section
   - You'll see different connection options

3. **Copy the Connection String**
   - Look for **URI** format (not JDBC or other formats)
   - Format: `postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres`
   
   **OR** use the **Direct connection** (port 5432):
   - Format: `postgresql://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres`
   
   **Your connection string** (you already have this):
   ```
   postgresql://postgres:NAgyxHiwctATJwro@db.lgbpmgaacqawvfavtdzu.supabase.co:5432/postgres
   ```

## Step 3: Verify Database Access

1. **Test Connection** (Optional)
   - You can test the connection using:
     - Supabase SQL Editor (built-in)
     - psql command line tool
     - Any PostgreSQL client

2. **Check Database Status**
   - In Supabase dashboard, go to **Database** → **Tables**
   - Should show empty (no tables yet - that's normal)
   - Tables will be created automatically when your app runs

## Step 4: Configure Database Settings (Optional)

### Enable Connection Pooling (Recommended)
- Supabase provides connection pooling automatically
- Use port **6543** for pooled connections (better for serverless)
- Use port **5432** for direct connections

### Set Up Database Extensions (If Needed)
- Go to **Database** → **Extensions**
- Most extensions are pre-installed
- Your app doesn't need any special extensions

## Step 5: Security Settings

1. **Check Network Access**
   - Go to **Settings** → **Database**
   - Under **Connection pooling**, ensure:
     - **Connection string** is set to allow connections
     - **IP restrictions** are not blocking Vercel

2. **Database Password**
   - Your password: `NAgyxHiwctATJwro`
   - Keep this secure!
   - You can reset it in **Settings** → **Database** → **Database password**

## Step 6: Understanding Your Connection String

Your connection string breakdown:
```
postgresql://postgres:NAgyxHiwctATJwro@db.lgbpmgaacqawvfavtdzu.supabase.co:5432/postgres
```

- **Protocol**: `postgresql://`
- **User**: `postgres`
- **Password**: `NAgyxHiwctATJwro`
- **Host**: `db.lgbpmgaacqawvfavtdzu.supabase.co`
- **Port**: `5432` (direct connection)
- **Database**: `postgres`

## Step 7: Alternative Connection String (Pooled)

For better performance with serverless (Vercel), you can use pooled connection:

**Pooled Connection** (port 6543):
```
postgresql://postgres.lgbpmgaacqawvfavtdzu:NAgyxHiwctATJwro@aws-0-us-east-1.pooler.supabase.com:6543/postgres
```

**Direct Connection** (port 5432) - What you're using:
```
postgresql://postgres:NAgyxHiwctATJwro@db.lgbpmgaacqawvfavtdzu.supabase.co:5432/postgres
```

Both work! Pooled is better for high traffic, direct is simpler.

## Step 8: Set Up in Vercel

Now that you have your Supabase database ready:

1. **Go to Vercel Dashboard**
   - https://vercel.com
   - Your project → **Settings** → **Environment Variables**

2. **Add DATABASE_URL**
   - Key: `DATABASE_URL`
   - Value: `postgresql://postgres:NAgyxHiwctATJwro@db.lgbpmgaacqawvfavtdzu.supabase.co:5432/postgres`
   - Environments: All (Production, Preview, Development)
   - Save

3. **Redeploy**
   - Your app will automatically create tables on first run

## Step 9: Verify Tables Are Created

After deploying, check Supabase:

1. **Go to Supabase Dashboard**
2. **Database** → **Tables**
3. You should see these tables:
   - `users` (for user authentication)
   - `memory_bank` (for user preferences)
   - Other tables your app creates

## Step 10: Monitor Database Usage

1. **Check Database Size**
   - **Settings** → **Database** → **Database size**
   - Free tier: 500 MB

2. **Monitor Connections**
   - **Settings** → **Database** → **Connection pooling**
   - Free tier: 60 direct connections, unlimited pooled

3. **View Logs**
   - **Logs** → **Postgres Logs**
   - See database queries and errors

## 🆘 Troubleshooting

### Connection Refused
- **Problem**: Can't connect to database
- **Fix**: 
  - Verify connection string is correct
  - Check password is correct
  - Ensure project is active (not paused)

### Authentication Failed
- **Problem**: Wrong password
- **Fix**: 
  - Reset password in **Settings** → **Database**
  - Update `DATABASE_URL` in Vercel

### Database Not Found
- **Problem**: Wrong database name
- **Fix**: Use `postgres` as database name (default)

### Tables Not Created
- **Problem**: App hasn't run yet
- **Fix**: 
  - Deploy your app
  - Make a request to trigger database initialization
  - Check Vercel logs for errors

## ✅ Checklist

- [ ] Supabase account created
- [ ] Project created and ready
- [ ] Database password saved securely
- [ ] Connection string copied
- [ ] `DATABASE_URL` set in Vercel
- [ ] App deployed
- [ ] Tables created in Supabase dashboard
- [ ] Login tested and working

---

**Your Supabase database is ready!** 🎉

The connection string you have is correct and will work once you set it in Vercel.

