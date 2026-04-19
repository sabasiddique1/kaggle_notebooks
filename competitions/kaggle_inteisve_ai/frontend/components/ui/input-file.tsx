"use client"

import * as React from "react"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { cn } from "@/lib/utils"

export interface InputFileProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
}

const InputFile = React.forwardRef<HTMLInputElement, InputFileProps>(
  ({ className, label, ...props }, ref) => {
    const inputId = React.useId()
    
    return (
      <div className="space-y-2">
        {label && <Label htmlFor={inputId}>{label}</Label>}
        <Input
          id={inputId}
          type="file"
          ref={ref}
          className={cn(
            "cursor-pointer file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90",
            className
          )}
          {...props}
        />
      </div>
    )
  }
)
InputFile.displayName = "InputFile"

export { InputFile }





