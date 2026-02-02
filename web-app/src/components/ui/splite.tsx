'use client'

import { Suspense, lazy, useRef, useEffect } from 'react'
// import Spline from '@splinetool/react-spline' // Direct import is often better for server components issues, but user used lazy. 
// User code:
const Spline = lazy(() => import('@splinetool/react-spline'))

interface SplineSceneProps {
    scene: string
    className?: string
}

export function SplineScene({ scene, className }: SplineSceneProps) {
    const splineRef = useRef<any>(null)

    const onLoad = (splineApp: any) => {
        splineRef.current = splineApp
    }

    useEffect(() => {
        const handleGlobalMouseMove = (e: MouseEvent) => {
            // Prevent infinite loop if we catch our own synthetic event
            if (!e.isTrusted) return

            if (splineRef.current && splineRef.current.canvas) {
                const canvas = splineRef.current.canvas
                const rect = canvas.getBoundingClientRect()

                // Check if mouse is strictly outside the canvas
                if (
                    e.clientX < rect.left ||
                    e.clientX > rect.right ||
                    e.clientY < rect.top ||
                    e.clientY > rect.bottom
                ) {
                    // Dispatch synthetic PointerEvent to canvas to force "Look At" update
                    // Spline usually relies on Pointer Events for cross-device support
                    const evt = new PointerEvent('pointermove', {
                        bubbles: true,
                        cancelable: true,
                        view: window,
                        clientX: e.clientX,
                        clientY: e.clientY,
                        pointerId: 1,
                        pointerType: 'mouse',
                        isPrimary: true,
                    })
                    canvas.dispatchEvent(evt)
                }
            }
        }

        window.addEventListener('mousemove', handleGlobalMouseMove)
        return () => window.removeEventListener('mousemove', handleGlobalMouseMove)
    }, [])

    return (
        <Suspense
            fallback={
                <div className="w-full h-full flex items-center justify-center">
                    <span className="loader"></span>
                </div>
            }
        >
            <Spline
                scene={scene}
                className={className}
                onLoad={onLoad}
            />
        </Suspense>
    )
}
