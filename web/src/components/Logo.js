import { motion } from 'framer-motion'
import Link from 'next/link'
import React from 'react'


let MotionLink = motion(Link);

const Logo = () => {

  return (
    <div
     className='flex flex-col items-center justify-center mt-2'>
        <MotionLink href="/" 
    className='flex items-center justify-center rounded-full w-16 h-16  bg-dark text-white dark:border-2 dark:border-solid dark:border-light
    text-xl font-bold'
    whileHover={{
      backgroundColor:["#121212", "rgba(182,62,150,1)","rgba(88,230,217,1)","rgba(182,62,150,1)", "#121212"],
      transition:{duration:1, repeat: Infinity }
    }}
    >NPC</MotionLink>
    </div>
  )
}

export default Logo