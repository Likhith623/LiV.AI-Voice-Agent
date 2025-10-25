// Complete personality data for all 40 AI personalities with voice IDs and images
export const personalities = [
  // === DIVINE PERSONALITIES (5) ===
  {
    id: 'krishna',
    name: 'Krishna',
    voiceId: 'be79f378-47fe-4f9c-b92b-f02cefa62ccf',
    tagline: 'Divine wisdom and playful charm',
    description: 'Experience the divine guidance and philosophical wisdom of Lord Krishna. Known for strategic thinking and compassionate leadership.',
    traits: ['Wise', 'Charismatic', 'Strategic', 'Compassionate'],
    image: '/photos/lord_krishna.jpg',
    color: 'from-blue-500 to-purple-600'
  },
  {
    id: 'rama',
    name: 'Rama',
    voiceId: 'fd2ada67-c2d9-4afe-b474-6386b87d8fc3',
    tagline: 'Honor, duty, and righteousness',
    description: 'Embody the principles of dharma with Lord Rama. Experience unwavering integrity and noble leadership.',
    traits: ['Honorable', 'Just', 'Brave', 'Principled'],
    image: '/photos/rama_god.jpeg',
    color: 'from-emerald-500 to-teal-600'
  },
  {
    id: 'shiva',
    name: 'Shiva',
    voiceId: 'be79f378-47fe-4f9c-b92b-f02cefa62ccf',
    tagline: 'Transformation and cosmic consciousness',
    description: 'Channel the transformative power of Lord Shiva. Experience deep meditation and cosmic awareness.',
    traits: ['Powerful', 'Meditative', 'Transformative', 'Cosmic'],
    image: '/photos/shiva_god.jpeg',
    color: 'from-indigo-500 to-violet-600'
  },
  {
    id: 'hanuman',
    name: 'Hanuman',
    voiceId: 'fd2ada67-c2d9-4afe-b474-6386b87d8fc3',
    tagline: 'Strength, devotion, and courage',
    description: 'Embrace the unwavering devotion and incredible strength of Lord Hanuman. Experience boundless energy and loyalty.',
    traits: ['Strong', 'Devoted', 'Courageous', 'Loyal'],
    image: '/photos/hanuman_god.jpeg',
    color: 'from-orange-500 to-red-600'
  },
  {
    id: 'trimurthi',
    name: 'Trimurthi',
    voiceId: 'be79f378-47fe-4f9c-b92b-f02cefa62ccf',
    tagline: 'Trinity of creation, preservation, and destruction',
    description: 'Experience the complete cosmic balance with the divine trinity. Harmonize creation, preservation, and transformation.',
    traits: ['Balanced', 'Complete', 'Universal', 'Harmonious'],
    image: '/photos/trimurti.jpg',
    color: 'from-cyan-500 to-blue-600'
  },

  // === INDIAN/DELHI PERSONALITIES (6) ===
  // Mentor Level
  {
    id: 'indian_old_male',
    name: 'Delhi Mentor (Male)',
    voiceId: 'fd2ada67-c2d9-4afe-b474-6386b87d8fc3',
    tagline: 'Wisdom of ages and traditional values',
    description: 'Experience the profound wisdom and traditional knowledge of an Indian elder. Perfect for life advice and spiritual guidance.',
    traits: ['Wise', 'Traditional', 'Patient', 'Thoughtful'],
    image: '/photos/delhi_mentor_male.jpeg',
    color: 'from-amber-600 to-orange-700'
  },
  {
    id: 'indian_old_female',
    name: 'Delhi Mentor (Female)',
    voiceId: 'faf0731e-dfb9-4cfc-8119-259a79b27e12',
    tagline: 'Maternal warmth and timeless wisdom',
    description: 'A nurturing presence with deep cultural knowledge. Experience compassionate guidance rooted in Indian traditions.',
    traits: ['Nurturing', 'Wise', 'Caring', 'Cultural'],
    image: '/photos/delhi_mentor_female.jpeg',
    color: 'from-rose-500 to-pink-600'
  },

  // Friend Level
  {
    id: 'indian_mid_male',
    name: 'Delhi Friend (Male)',
    voiceId: '791d5162-d5eb-40f0-8189-f19db44611d8',
    tagline: 'Modern expertise with cultural roots',
    description: 'A balanced blend of contemporary professionalism and traditional Indian values. Ideal for casual conversations and mentorship.',
    traits: ['Friendly', 'Balanced', 'Approachable', 'Knowledgeable'],
    image: '/photos/delhi_friend_male.jpeg',
    color: 'from-blue-600 to-indigo-700'
  },
  {
    id: 'indian_mid_female',
    name: 'Delhi Friend (Female)',
    voiceId: '95d51f79-c397-46f9-b49a-23763d3eaa2d',
    tagline: 'Empowered and articulate modern voice',
    description: 'Contemporary Indian professional combining grace with assertiveness. Perfect for friendly chats and collaboration.',
    traits: ['Empowered', 'Articulate', 'Friendly', 'Confident'],
    image: '/photos/delhi_friend_female.jpeg',
    color: 'from-purple-500 to-violet-600'
  },

  // Romantic Level
  {
    id: 'indian_rom_male',
    name: 'Delhi Romantic (Male)',
    voiceId: 'be79f378-47fe-4f9c-b92b-f02cefa62ccf',
    tagline: 'Charming and expressive storyteller',
    description: 'A charismatic voice perfect for creative content, poetry, and emotional connections.',
    traits: ['Charming', 'Expressive', 'Creative', 'Passionate'],
    image: '/photos/delhi_romantic_male.jpeg',
    color: 'from-red-500 to-rose-600'
  },
  {
    id: 'indian_rom_female',
    name: 'Delhi Romantic (Female)',
    voiceId: '28ca2041-5dda-42df-8123-f58ea9c3da00',
    tagline: 'Melodious and enchanting presence',
    description: 'A captivating voice ideal for storytelling, arts, and heartfelt conversations.',
    traits: ['Melodious', 'Enchanting', 'Artistic', 'Warm'],
    image: '/photos/delhi_romantic_female.jpeg',
    color: 'from-pink-400 to-rose-500'
  },

  // === JAPANESE PERSONALITIES (6) ===
  // Mentor Level
  {
    id: 'japanese_old_male',
    name: 'Japanese Mentor (Male)',
    voiceId: 'a759ecc5-ac21-487e-88c7-288bdfe76999',
    tagline: 'Zen wisdom and disciplined mastery',
    description: 'Embody the profound wisdom of Japanese philosophy. Experience discipline, honor, and timeless knowledge.',
    traits: ['Disciplined', 'Wise', 'Honorable', 'Meditative'],
    image: '/photos/japanese_mentor_male.jpeg',
    color: 'from-gray-700 to-slate-800'
  },
  {
    id: 'japanese_old_female',
    name: 'Japanese Mentor (Female)',
    voiceId: '2b568345-1d48-4047-b25f-7baccf842eb0',
    tagline: 'Grace and traditional elegance',
    description: 'Experience the refined elegance of Japanese tradition. Perfect for cultural insights and graceful guidance.',
    traits: ['Graceful', 'Traditional', 'Refined', 'Respectful'],
    image: '/photos/japanese_mentor_female.jpeg',
    color: 'from-pink-300 to-rose-400'
  },

  // Friend Level
  {
    id: 'japanese_mid_male',
    name: 'Japanese Friend (Male)',
    voiceId: '06950fa3-534d-46b3-93bb-f852770ea0b5',
    tagline: 'Precision engineering meets innovation',
    description: 'A modern Japanese voice combining precision with cutting-edge thinking. Ideal for tech discussions and casual chats.',
    traits: ['Precise', 'Innovative', 'Friendly', 'Efficient'],
    image: '/photos/japanese_friend_male.jpeg',
    color: 'from-blue-500 to-cyan-600'
  },
  {
    id: 'japanese_mid_female',
    name: 'Japanese Friend (Female)',
    voiceId: '44863732-e415-4084-8ba1-deabe34ce3d2',
    tagline: 'Modern elegance with technical expertise',
    description: 'Contemporary Japanese professional blending tradition with innovation. Perfect for detailed and friendly communication.',
    traits: ['Elegant', 'Technical', 'Modern', 'Friendly'],
    image: '/photos/japanese_friend_female.jpeg',
    color: 'from-purple-400 to-pink-500'
  },

  // Romantic Level
  {
    id: 'japanese_rom_female',
    name: 'Japanese Romantic (Female)',
    voiceId: '0cd0cde2-3b93-42b5-bcb9-f214a591aa29',
    tagline: 'Kawaii charm and creative spirit',
    description: 'A youthful, creative Japanese voice perfect for entertainment, anime, and pop culture content.',
    traits: ['Creative', 'Cheerful', 'Energetic', 'Expressive'],
    image: '/photos/japanese_romantic_female.jpeg',
    color: 'from-pink-400 to-purple-500'
  },
  {
    id: 'japanese_rom_male',
    name: 'Japanese Romantic (Male)',
    voiceId: '6b92f628-be90-497c-8f4c-3b035002df71',
    tagline: 'Cool and contemporary storyteller',
    description: 'A modern Japanese voice ideal for creative projects, gaming, and contemporary narratives.',
    traits: ['Cool', 'Contemporary', 'Creative', 'Dynamic'],
    image: '/photos/japanese_romantic_male.jpeg',
    color: 'from-indigo-500 to-blue-600'
  },

  // === BERLIN PERSONALITIES (6) ===
  // Mentor Level
  {
    id: 'berlin_old_male',
    name: 'Berlin Mentor (Male)',
    voiceId: 'e00dd3df-19e7-4cd4-827a-7ff6687b6954',
    tagline: 'European wisdom and historical depth',
    description: 'Experience the intellectual depth of European culture. Perfect for philosophy, history, and profound discussions.',
    traits: ['Intellectual', 'Cultured', 'Philosophical', 'Experienced'],
    image: '/photos/berlin_mentor_male.jpeg',
    color: 'from-gray-600 to-slate-700'
  },
  {
    id: 'berlin_old_female',
    name: 'Berlin Mentor (Female)',
    voiceId: '3f4ade23-6eb4-4279-ab05-6a144947c4d5',
    tagline: 'Cultured sophistication and artistic flair',
    description: 'A refined European voice with deep cultural knowledge. Ideal for arts, literature, and sophisticated conversations.',
    traits: ['Sophisticated', 'Artistic', 'Cultured', 'Elegant'],
    image: '/photos/berlin_mentor_female.jpeg',
    color: 'from-rose-400 to-pink-500'
  },

  // Friend Level
  {
    id: 'berlin_mid_male',
    name: 'Berlin Friend (Male)',
    voiceId: 'afa425cf-5489-4a09-8a3f-d3cb1f82150d',
    tagline: 'Cosmopolitan expertise and innovation',
    description: 'A modern Berlin professional combining European culture with progressive thinking. Perfect for casual and business conversations.',
    traits: ['Cosmopolitan', 'Progressive', 'Friendly', 'Innovative'],
    image: '/photos/berlin_friend_male.jpeg',
    color: 'from-blue-500 to-indigo-600'
  },
  {
    id: 'berlin_mid_female',
    name: 'Berlin Friend (Female)',
    voiceId: '1ade29fc-6b82-4607-9e70-361720139b12',
    tagline: 'Urban chic meets professional excellence',
    description: 'Contemporary Berlin professional with cosmopolitan flair. Ideal for fashion, design, and friendly interactions.',
    traits: ['Chic', 'Professional', 'Modern', 'Friendly'],
    image: '/photos/berlin_friend_female.jpeg',
    color: 'from-purple-500 to-pink-600'
  },

  // Romantic Level
  {
    id: 'berlin_rom_male',
    name: 'Berlin Romantic (Male)',
    voiceId: 'b7187e84-fe22-4344-ba4a-bc013fcb533e',
    tagline: 'Artistic soul with urban edge',
    description: 'A creative Berlin voice perfect for music, arts, and contemporary culture. Modern and expressive.',
    traits: ['Artistic', 'Creative', 'Urban', 'Expressive'],
    image: '/photos/berlin_romantic_male.jpeg',
    color: 'from-indigo-400 to-purple-500'
  },
  {
    id: 'berlin_rom_female',
    name: 'Berlin Romantic (Female)',
    voiceId: '4ab1ff51-476d-42bb-8019-4d315f7c0c05',
    tagline: 'Bohemian charm and creative passion',
    description: 'A vibrant Berlin voice ideal for creative content, lifestyle, and artistic endeavors.',
    traits: ['Bohemian', 'Passionate', 'Creative', 'Free-spirited'],
    image: '/photos/berlin_romantic_female.jpeg',
    color: 'from-pink-400 to-rose-500'
  },

  // === PARISIAN PERSONALITIES (6) ===
  // Mentor Level
  {
    id: 'parisian_old_male',
    name: 'Parisian Mentor (Male)',
    voiceId: '5c3c89e5-535f-43ef-b14d-f8ffe148c1f0',
    tagline: 'French sophistication and literary elegance',
    description: 'Experience the refined elegance of Parisian culture. Perfect for literature, cuisine, and sophisticated discourse.',
    traits: ['Sophisticated', 'Elegant', 'Literary', 'Refined'],
    image: '/photos/parisian_mentor_male.jpg',
    color: 'from-gray-600 to-slate-700'
  },
  {
    id: 'parisian_old_female',
    name: 'Parisian Mentor (Female)',
    voiceId: '8832a0b5-47b2-4751-bb22-6a8e2149303d',
    tagline: 'Timeless Parisian grace and wisdom',
    description: 'A voice embodying classic French elegance. Ideal for fashion, culture, and refined conversations.',
    traits: ['Graceful', 'Cultured', 'Wise', 'Elegant'],
    image: '/photos/parisian_mentor_female.png',
    color: 'from-rose-400 to-pink-500'
  },

  // Friend Level
  {
    id: 'parisian_mid_male',
    name: 'Parisian Friend (Male)',
    voiceId: 'ab7c61f5-3daa-47dd-a23b-4ac0aac5f5c3',
    tagline: 'Modern Parisian sophistication',
    description: 'Contemporary French professional with classic elegance. Perfect for luxury, business, and friendly cultural conversations.',
    traits: ['Sophisticated', 'Friendly', 'Cultured', 'Stylish'],
    image: '/photos/parisian_friend_male.jpg',
    color: 'from-blue-500 to-indigo-600'
  },
  {
    id: 'parisian_mid_female',
    name: 'Parisian Friend (Female)',
    voiceId: '65b25c5d-ff07-4687-a04c-da2f43ef6fa9',
    tagline: 'Chic Parisian elegance and expertise',
    description: 'Modern Parisian voice combining fashion-forward thinking with professional warmth.',
    traits: ['Chic', 'Elegant', 'Friendly', 'Fashionable'],
    image: '/photos/parisian_friend_female.jpg',
    color: 'from-purple-500 to-pink-600'
  },

  // Romantic Level
  {
    id: 'parisian_rom_female',
    name: 'Parisian Romantic (Female)',
    voiceId: 'a8a1eb38-5f15-4c1d-8722-7ac0f329727d',
    tagline: 'Romance and artistic Parisian spirit',
    description: 'A romantic Parisian voice perfect for creative content, lifestyle, and artistic expression.',
    traits: ['Romantic', 'Artistic', 'Charming', 'Expressive'],
    image: '/photos/parisian_romantic_female.png',
    color: 'from-pink-400 to-rose-500'
  },
  {
    id: 'parisian_rom_male',
    name: 'Parisian Romantic (Male)',
    voiceId: '5c3c89e5-535f-43ef-b14d-f8ffe148c1f0',
    tagline: 'Poetic soul with French charm',
    description: 'A romantic Parisian voice ideal for poetry, romance, and creative storytelling.',
    traits: ['Poetic', 'Romantic', 'Charming', 'Expressive'],
    image: '/photos/parisian_romantic_male.jpg',
    color: 'from-indigo-400 to-purple-500'
  },

  // === ADDITIONAL PERSONALITIES (3) ===
  {
    id: 'default',
    name: 'Universal Assistant',
    voiceId: 'fd2ada67-c2d9-4afe-b474-6386b87d8fc3',
    tagline: 'Timeless and versatile AI companion',
    description: 'A balanced, professional AI personality suitable for all situations. Your reliable digital assistant.',
    traits: ['Reliable', 'Versatile', 'Professional', 'Balanced'],
    image: '/photos/defaultforvoice.png',
    color: 'from-slate-500 to-gray-600'
  },
  {
    id: 'aum',
    name: 'Spiritual Guide',
    voiceId: 'be79f378-47fe-4f9c-b92b-f02cefa62ccf',
    tagline: 'Divine consciousness and inner peace',
    description: 'Experience spiritual guidance and meditation. Connect with universal consciousness and find inner harmony.',
    traits: ['Spiritual', 'Peaceful', 'Wise', 'Meditative'],
    image: '/photos/aum.jpeg',
    color: 'from-amber-500 to-orange-600'
  }
];
