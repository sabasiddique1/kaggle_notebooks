# Database Setup Guide

## 🎯 Problem Solved

**SQLite on Vercel is unreliable** - files in `/tmp` are ephemeral and get wiped on cold starts.

**Solution**: Use **PostgreSQL** in production (works perfectly on Vercel serverless).

## 🚀 Quick Setup Options

### Option 1: Supabase (Recommended - Free & Easy)

1. **Create Supabase Account**: https://supabase.com (free tier)
2. **Create New Project**
3. **Get Connection String**:
   - Go to Project Settings → Database
   - Copy "Connection string" (URI format)
   - Format: `postgresql://postgres:[PASSWORD]@[HOST]:5432/postgres`

4. **Set in Vercel**:
   - Go to Vercel Dashboard → Your Project → Settings → Environment Variables
   - Add: `DATABASE_URL` = `postgresql://postgres:...`

5. **Deploy** - That's it! ✅

### Option 2: Vercel Postgres

1. **Add Vercel Postgres**:
   - Vercel Dashboard → Your Project → Storage → Create Database
   - Select "Postgres"
   - Vercel will automatically set `DATABASE_URL`

2. **Deploy** - That's it! ✅

### Option 3: Neon (Serverless Postgres)

1. **Create Neon Account**: https://neon.tech (free tier)
2. **Create Project**
3. **Get Connection String**
4. **Set in Vercel**: `DATABASE_URL` environment variable

## 📋 Current Configuration

The code now:
- ✅ **Automatically detects** if `DATABASE_URL` is set
- ✅ **Uses PostgreSQL** if `DATABASE_URL` is provided
- ✅ **Falls back to SQLite** for local development
- ✅ **Works with any PostgreSQL provider** (Supabase, Vercel Postgres, Neon, etc.)

## 🔧 Environment Variables

**Required for Production:**
```
DATABASE_URL=postgresql://user:password@host:port/database
JWT_SECRET_KEY=your-secret-key
```

**Optional:**
```
GEMINI_API_KEY=your-key
```

## ✅ After Setup

1. Deploy to Vercel
2. Database tables will be created automatically
3. Demo users will be initialized automatically
4. Login/Register will work reliably! ✅

## 🧪 Testing

After deployment, test:
- Login: `patient@demo.com` / `patient123`
- Register: Create new account
- Check `/api/debug/auth` for database status

---

**No more SQLite issues! PostgreSQL works perfectly on Vercel serverless!** 🎉

