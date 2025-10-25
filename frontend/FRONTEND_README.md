# LiV.AI Voice Agent - Frontend

A modern, premium-quality AI personality selector built with React and Tailwind CSS.

## 🎨 Features

- **Premium Design**: Dark theme with glassmorphism effects, gradient accents, and smooth animations
- **Interactive UI**: Smooth hover effects, card animations, and modal interactions
- **Responsive Layout**: Fully responsive design for desktop, tablet, and mobile devices
- **9+ AI Personalities**: Choose from diverse personalities including Krishna, Rama, Shiva, Hanuman, and international assistants
- **Framer Motion Animations**: Silky-smooth transitions and micro-interactions
- **Clean Typography**: Modern sans-serif fonts (Inter, Poppins) for excellent readability

## 🚀 Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open your browser and visit:
   ```
   http://localhost:3000
   ```

## 📁 Project Structure

```
frontend/
├── public/
│   └── photos/              # Personality images
├── src/
│   ├── components/
│   │   ├── Navbar.jsx       # Top navigation bar
│   │   ├── Footer.jsx       # Footer with social links
│   │   └── PersonalityCard.jsx  # Card & Modal components
│   ├── data/
│   │   └── personalities.js # Personality data
│   ├── App.js               # Main application
│   └── index.css            # Global styles & Tailwind
├── tailwind.config.js       # Tailwind configuration
└── package.json
```

## 🎨 Design System

### Colors
- **Primary**: Blue gradient (#3b82f6 to #8b5cf6)
- **Accent**: Violet (#8b5cf6), Teal (#14b8a6), Cyan (#06b6d4)
- **Background**: Dark navy/black with gradients

### Typography
- **Headings**: Poppins/Inter Bold (700-800)
- **Body**: Inter Regular (400)
- **Accents**: Inter Medium (500-600)

### Effects
- Glassmorphism backgrounds
- Soft neon glows
- Smooth scale & float animations
- Backdrop blur effects

## 🛠️ Technologies Used

- **React**: UI library
- **Tailwind CSS**: Utility-first CSS framework
- **Framer Motion**: Animation library
- **React Icons**: Icon library

## 📦 Build for Production

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## 🎯 Available Personalities

1. **Krishna** - Divine wisdom and playful charm
2. **Rama** - Honor, duty, and righteousness
3. **Shiva** - Transformation and cosmic consciousness
4. **Hanuman** - Strength, devotion, and courage
5. **Trimurthi** - Trinity of creation, preservation, and destruction
6. **Berlin Assistant** - Sophisticated European elegance
7. **Tokyo Assistant** - Precision meets elegance
8. **Delhi Mentor** - Warmth and wisdom from the heart of India
9. **Classic Assistant** - Timeless and versatile AI companion

## 🎨 Customization

### Adding New Personalities

Edit `src/data/personalities.js` and add a new personality object:

```javascript
{
  id: 'unique-id',
  name: 'Personality Name',
  tagline: 'Short tagline',
  description: 'Detailed description',
  traits: ['Trait1', 'Trait2', 'Trait3', 'Trait4'],
  image: '/photos/image_name.png',
  color: 'from-blue-500 to-purple-600'
}
```

### Modifying Colors

Edit `tailwind.config.js` to customize the color palette.

## 📄 License

MIT License - feel free to use this project for your own purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Made with ❤️ by LiV.AI Team
