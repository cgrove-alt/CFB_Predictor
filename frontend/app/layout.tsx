import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Sharp Picks CFB',
  description: 'College Football Spread Predictions',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-900">
        {children}
      </body>
    </html>
  )
}
