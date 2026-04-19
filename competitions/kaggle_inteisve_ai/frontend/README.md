# EmergencyCareNavigator Frontend

A professional Next.js frontend for the EmergencyCareNavigator multi-agent emergency care system.

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Components**: shadcn/ui
- **Icons**: lucide-react

## Getting Started

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── app/                    # Next.js app directory
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Dashboard
│   ├── triage/            # Triage workspace
│   ├── sessions/          # Sessions history
│   ├── logs/              # Observability logs
│   └── settings/          # Settings page
├── components/
│   ├── ui/                # shadcn/ui components
│   ├── layout/            # App shell, sidebar, header
│   ├── triage/            # Triage components
│   ├── facilities/        # Facility components
│   ├── sbar/              # SBAR handoff panel
│   └── booking/           # Booking dialog
└── lib/
    ├── utils.ts           # Utility functions
    └── api.ts             # API client
```

## Features

- ✅ Responsive design (desktop-first, mobile-friendly)
- ✅ Modern medical dashboard UI
- ✅ Complete triage workflow
- ✅ Facility search and selection
- ✅ SBAR handoff packet display
- ✅ Booking workflow
- ✅ Sessions history
- ✅ Observability logs
- ✅ Settings management

## API Integration

The frontend connects to the FastAPI backend running on `http://localhost:8000`. Update `NEXT_PUBLIC_API_URL` in `.env.local` if needed.

## Design System

- **Colors**: Medical teal primary, red for emergencies, amber for warnings
- **Typography**: Inter font family
- **Components**: Rounded cards, subtle shadows, clean spacing
- **Icons**: lucide-react (no emojis)

## License

MIT





