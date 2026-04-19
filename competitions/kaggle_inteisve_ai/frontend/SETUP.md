# Frontend Setup Guide

## Quick Start

1. **Install Dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env.local
   # Edit .env.local if your API runs on a different port
   ```

3. **Start Development Server**
   ```bash
   npm run dev
   ```

4. **Open Browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

## Prerequisites

- Node.js 18+ and npm
- Backend API running on http://localhost:8000 (or update NEXT_PUBLIC_API_URL)

## Project Structure

```
frontend/
├── app/                    # Next.js pages
│   ├── page.tsx           # Dashboard (/)
│   ├── triage/            # Triage workspace (/triage)
│   ├── sessions/          # Sessions history (/sessions)
│   ├── logs/              # Observability logs (/logs)
│   └── settings/          # Settings (/settings)
├── components/
│   ├── ui/                # shadcn/ui base components
│   ├── layout/            # App shell, sidebar, header
│   ├── triage/            # Triage-specific components
│   ├── facilities/        # Facility components
│   ├── sbar/              # SBAR handoff panel
│   └── booking/           # Booking dialog
└── lib/
    ├── utils.ts           # Utility functions (cn helper)
    └── api.ts             # API client functions
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint

## Features Implemented

✅ **Dashboard** - Stats, recent activity, quick start
✅ **Triage Workspace** - 3-column layout with intake, results, facilities
✅ **SBAR Panel** - Clinician-ready handoff packet
✅ **Booking Dialog** - Facility booking workflow
✅ **Sessions Page** - Historical triage sessions
✅ **Logs Page** - Observability metrics and event logs
✅ **Settings Page** - Configuration management
✅ **Responsive Design** - Mobile-friendly layout
✅ **Modern UI** - shadcn/ui components, Tailwind CSS

## Design Notes

- **No Emojis**: All icons use lucide-react
- **Medical Theme**: Calm, professional, healthcare-oriented
- **Color Palette**: Medical teal primary, red for emergencies
- **Typography**: Inter font, clear hierarchy
- **Components**: Rounded cards, subtle shadows, clean spacing

## Troubleshooting

### Port Already in Use
Change the port in `package.json`:
```json
"dev": "next dev -p 3001"
```

### API Connection Issues
1. Ensure backend is running on http://localhost:8000
2. Check CORS settings in backend
3. Verify NEXT_PUBLIC_API_URL in .env.local

### Build Errors
1. Clear `.next` folder: `rm -rf .next`
2. Reinstall dependencies: `rm -rf node_modules && npm install`
3. Check TypeScript errors: `npm run build`

## Next Steps

- Add dark mode support
- Implement real-time updates
- Add more visualizations
- Enhance mobile experience
- Add unit tests





