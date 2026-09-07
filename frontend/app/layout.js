import "./globals.css";

export const metadata = {
  title: "Camera Software",
  description: "Camera inference and model operations console",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
